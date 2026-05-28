//! Bench: реальный prod-аудио прогон Parakeet на Metal с замерами.
//!
//! Запуск:
//!   cd RustASR
//!   cargo run --release --example bench_metal -p model-parakeet --features candle-core/metal
//!
//! Аудио: /Users/askid/Library/Application Support/com.yttri.app/users/.../full.mp3
//! (4-минутная prod запись с микрофона; берём из неё чанки разной длительности).

use std::path::PathBuf;
use std::time::Instant;

use asr_core::{AsrModel, TranscribeOptions};
use candle_core::Device;
use model_parakeet::ParakeetModel;

const PROD_AUDIO_PATHS: &[&str] = &[
    "/Users/askid/Library/Application Support/com.yttri.app/users/askid.mobile_at_gmail.com/recordings/584ce801-f009-46ea-a4dc-aec7fabc4429/full.mp3",
    "/Users/askid/Library/Application Support/com.yttri.app/users/askid.mobile_at_gmail.com/recordings/ae455068-5ee9-43aa-998f-c4b2ea4f643a/full.mp3",
];

const TEST_WAV_PATHS: &[&str] = &[
    "/Volumes/Askid Dev/Projects/RustASR/test_prod_3s.wav",
    "/Volumes/Askid Dev/Projects/RustASR/test_real_audio.wav",
];

const MODEL_PATH: &str = "/Volumes/Askid Dev/Projects/RustASR/models/parakeet-tdt-0.6b-v3";

/// macOS phys_footprint via `footprint` — единственный честный показатель.
/// Считает: anonymous heap + mmap'd file pages touched + Metal unified memory.
/// `ps -o rss=` НЕ включает Metal GPU buffers (отдельный аллокатор Apple Silicon).
fn process_phys_footprint_mb() -> Option<u64> {
    #[cfg(target_os = "macos")]
    {
        // `footprint` CLI: первая строка "<name> [pid]: 64-bit    Footprint: NNNN KB"
        let pid = std::process::id().to_string();
        let output = std::process::Command::new("footprint")
            .args(["-p", &pid])
            .output()
            .ok()?;
        let s = String::from_utf8_lossy(&output.stdout);
        // Parse "Footprint: NNNN KB" anywhere in stdout
        for line in s.lines() {
            if let Some(idx) = line.find("Footprint:") {
                let tail = &line[idx + "Footprint:".len()..];
                let parts: Vec<&str> = tail.trim().split_whitespace().collect();
                if parts.len() >= 2 {
                    let num: f64 = parts[0].parse().ok()?;
                    let unit = parts[1];
                    let mb = match unit {
                        "B" => num / 1_048_576.0,
                        "KB" => num / 1024.0,
                        "MB" => num,
                        "GB" => num * 1024.0,
                        _ => return None,
                    };
                    return Some(mb as u64);
                }
            }
        }
        None
    }
    #[cfg(not(target_os = "macos"))]
    None
}

/// Fallback: ps -o rss= (НЕ включает Metal GPU memory!).
fn process_rss_mb() -> Option<u64> {
    #[cfg(target_os = "macos")]
    {
        let output = std::process::Command::new("ps")
            .args(["-o", "rss=", "-p", &std::process::id().to_string()])
            .output()
            .ok()?;
        let s = String::from_utf8_lossy(&output.stdout);
        let kb: u64 = s.trim().parse().ok()?;
        Some(kb / 1024)
    }
    #[cfg(not(target_os = "macos"))]
    None
}

/// vmmap parse: "REGION TYPE ... VIRTUAL SIZE DIRTY SIZE" — берём Metal regions.
fn metal_alloc_mb() -> Option<u64> {
    #[cfg(target_os = "macos")]
    {
        let pid = std::process::id().to_string();
        let output = std::process::Command::new("vmmap")
            .args(["-summary", &pid])
            .output()
            .ok()?;
        let s = String::from_utf8_lossy(&output.stdout);
        // Parse "IOKit                            X.XG  ..." и "Stack" "MALLOC_LARGE" etc.
        // На Apple Silicon Metal buffers идут как IOKit или Stack regions.
        let mut iokit_mb = 0u64;
        for line in s.lines() {
            // Грубо: ищем строки начинающиеся с "IOKit" или "VM_ALLOCATE" большие
            let lower = line.to_lowercase();
            if lower.contains("iokit") || lower.contains("mapped file") || lower.contains("vm_allocate") {
                // Парсим размер: формат "  IOKit    1.2G 1024K  ..."
                let parts: Vec<&str> = line.split_whitespace().collect();
                for p in &parts {
                    if let Some(num_str) = p.strip_suffix('G') {
                        if let Ok(gb) = num_str.parse::<f64>() {
                            iokit_mb += (gb * 1024.0) as u64;
                            break;
                        }
                    } else if let Some(num_str) = p.strip_suffix('M') {
                        if let Ok(mb) = num_str.parse::<f64>() {
                            iokit_mb += mb as u64;
                            break;
                        }
                    }
                }
            }
        }
        Some(iokit_mb)
    }
    #[cfg(not(target_os = "macos"))]
    None
}

/// Track peak memory by sampling in background thread.
struct PeakMemTracker {
    stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
    peak_phys_mb: std::sync::Arc<std::sync::atomic::AtomicU64>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl PeakMemTracker {
    fn start() -> Self {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
        let stop = Arc::new(AtomicBool::new(false));
        let peak = Arc::new(AtomicU64::new(0));
        let stop_clone = stop.clone();
        let peak_clone = peak.clone();
        let handle = std::thread::spawn(move || {
            while !stop_clone.load(Ordering::Relaxed) {
                if let Some(mb) = process_phys_footprint_mb() {
                    let prev = peak_clone.load(Ordering::Relaxed);
                    if mb > prev {
                        peak_clone.store(mb, Ordering::Relaxed);
                    }
                }
                std::thread::sleep(std::time::Duration::from_millis(50));
            }
        });
        Self { stop, peak_phys_mb: peak, handle: Some(handle) }
    }

    fn stop(mut self) -> u64 {
        self.stop.store(true, std::sync::atomic::Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
        self.peak_phys_mb.load(std::sync::atomic::Ordering::Relaxed)
    }
}

fn load_wav_16k(path: &str) -> Vec<f32> {
    let data = std::fs::read(path).expect("read wav");
    // WAV: 44 byte header, i16 PCM
    let pcm = &data[44..];
    pcm.chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0)
        .collect()
}

fn find_test_wav() -> Option<String> {
    for p in TEST_WAV_PATHS {
        if std::path::Path::new(p).exists() {
            return Some(p.to_string());
        }
    }
    None
}

fn extract_chunk_from_prod_mp3(out_path: &str, start_sec: u64, duration_sec: u64) -> Option<String> {
    // Берём первый существующий mp3 из списка и через ffmpeg извлекаем фрагмент.
    let src = PROD_AUDIO_PATHS
        .iter()
        .find(|p| std::path::Path::new(p).exists())?;
    let status = std::process::Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            src,
            "-ss",
            &start_sec.to_string(),
            "-t",
            &duration_sec.to_string(),
            "-ar",
            "16000",
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            out_path,
        ])
        .status()
        .ok()?;
    if !status.success() {
        return None;
    }
    Some(out_path.to_string())
}

fn default_opts() -> TranscribeOptions {
    TranscribeOptions {
        language: None,
        max_tokens: None,
        timestamps: true,
        temperature: 0.0,
    }
}

struct BenchResult {
    label: String,
    duration_sec: f64,
    inference_ms: u128,
    rtf: f64,
    text: String,
    word_count: usize,
    streaming_token_count: usize,
    peak_phys_mb: u64,
    delta_phys_mb: i64,
}

fn print_table(rows: &[BenchResult]) {
    println!();
    println!("┌──────────────────────┬───────┬──────┬───────┬────────┬───────┬───────────┬──────────┐");
    println!("│ label                │ audio │ ms   │ RTF   │ stream │ words │ peak phys │ delta    │");
    println!("│                      │ sec   │      │       │ tokens │       │ (МБ)      │ (МБ)     │");
    println!("├──────────────────────┼───────┼──────┼───────┼────────┼───────┼───────────┼──────────┤");
    for r in rows {
        println!(
            "│ {:<20} │ {:>5.2} │ {:>4} │ {:>5.3} │ {:>6} │ {:>5} │ {:>9} │ {:>+8} │",
            truncate(&r.label, 20),
            r.duration_sec,
            r.inference_ms,
            r.rtf,
            r.streaming_token_count,
            r.word_count,
            r.peak_phys_mb,
            r.delta_phys_mb
        );
    }
    println!("└──────────────────────┴───────┴──────┴───────┴────────┴───────┴───────────┴──────────┘");
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        s.chars().take(max - 1).collect::<String>() + "…"
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Tracing -> stderr (debug logs если RUST_LOG=debug).
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info,model_parakeet=info")),
        )
        .try_init();

    println!("=== Parakeet TDT 0.6B v3 — Metal bench ===");
    println!();

    let rss_before = process_rss_mb().unwrap_or(0);
    let phys_before = process_phys_footprint_mb().unwrap_or(0);
    let metal_before = metal_alloc_mb().unwrap_or(0);
    println!("BASELINE (до загрузки модели):");
    println!("  RSS:             {} МБ (ps -o rss=)", rss_before);
    println!("  Phys footprint:  {} МБ (footprint — включает Metal!)", phys_before);
    println!("  IOKit/Metal:     {} МБ (vmmap)", metal_before);

    let device = Device::new_metal(0).map_err(|e| format!("Metal init: {e}"))?;
    println!("Device: Metal GPU (M-series)");

    let load_t0 = Instant::now();
    let mut model = ParakeetModel::load(MODEL_PATH, &device)?;
    let load_ms = load_t0.elapsed().as_millis();

    let rss_after = process_rss_mb().unwrap_or(0);
    let phys_after = process_phys_footprint_mb().unwrap_or(0);
    let metal_after = metal_alloc_mb().unwrap_or(0);
    println!();
    println!("AFTER LOAD ({} мс):", load_ms);
    println!("  RSS:             {} МБ (+{} МБ)", rss_after, rss_after.saturating_sub(rss_before));
    println!("  Phys footprint:  {} МБ (+{} МБ) ← ЧЕСТНАЯ ЦИФРА", phys_after, phys_after.saturating_sub(phys_before));
    println!("  IOKit/Metal:     {} МБ (+{} МБ)", metal_after, metal_after.saturating_sub(metal_before));

    // Warmup: первая инференция всегда дольше (Metal kernel compile + driver setup).
    println!();
    println!("--- Warmup (1с тишины) ---");
    let warmup_samples = vec![0.0f32; 16000];
    let t0 = Instant::now();
    let _ = model.transcribe(&warmup_samples, &default_opts())?;
    let warmup_ms = t0.elapsed().as_millis();
    println!("Warmup: {} мс", warmup_ms);

    // Тесты: разные длительности из prod аудио.
    println!();
    println!("--- Бенч: реальное аудио (prod recording) ---");

    let mut results = Vec::new();

    // Извлекаем чанки разной длительности из prod MP3.
    let chunks: Vec<(&str, u64, u64)> = vec![
        ("prod 1.5s @ 60s", 60, 2),
        ("prod 3s @ 60s", 60, 3),
        ("prod 5s @ 60s", 60, 5),
        ("prod 10s @ 60s", 60, 10),
        ("prod 30s @ 30s", 30, 30),
    ];

    for (label, start_sec, dur_sec) in chunks {
        let tmp = format!("/tmp/parakeet_bench_{}_{}s.wav", start_sec, dur_sec);
        let path = match extract_chunk_from_prod_mp3(&tmp, start_sec, dur_sec) {
            Some(p) => p,
            None => {
                eprintln!("SKIP {}: no prod mp3 or ffmpeg failed", label);
                continue;
            }
        };

        let samples = load_wav_16k(&path);
        let actual_dur = samples.len() as f64 / 16000.0;

        // Замер streaming + peak memory во время transcribe.
        use std::cell::Cell;
        let stream_tokens = Cell::new(0usize);
        let phys_before_call = process_phys_footprint_mb().unwrap_or(0);
        let tracker = PeakMemTracker::start();

        let t0 = Instant::now();
        let result = model
            .transcribe_streaming(&samples, &default_opts(), |_text, _is_wb, _ts_ms| {
                stream_tokens.set(stream_tokens.get() + 1);
            })?;
        let elapsed_ms = t0.elapsed().as_millis();

        let peak_phys = tracker.stop();
        let phys_after_call = process_phys_footprint_mb().unwrap_or(0);
        let stream_tokens = stream_tokens.get();
        let delta = phys_after_call as i64 - phys_before_call as i64;

        let rtf = elapsed_ms as f64 / 1000.0 / actual_dur;
        let word_count = result.text.split_whitespace().count();

        println!(
            "[{}] {:.2}с → \"{}\" ({} мс, RTF={:.3}, {} stream tokens, {} words, peak={} МБ delta={:+} МБ)",
            label,
            actual_dur,
            truncate(&result.text, 80),
            elapsed_ms,
            rtf,
            stream_tokens,
            word_count,
            peak_phys,
            delta,
        );

        results.push(BenchResult {
            label: label.to_string(),
            duration_sec: actual_dur,
            inference_ms: elapsed_ms,
            rtf,
            text: result.text,
            word_count,
            streaming_token_count: stream_tokens,
            peak_phys_mb: peak_phys,
            delta_phys_mb: delta,
        });

        // Очистка temp.
        let _ = std::fs::remove_file(&path);
    }

    // Fallback на статичный test_real_audio.wav если prod недоступно.
    if results.is_empty() {
        if let Some(wav) = find_test_wav() {
            println!("Fallback: используем {}", wav);
            let samples = load_wav_16k(&wav);
            let chunk_3s = &samples[..(3 * 16000).min(samples.len())];
            let actual_dur = chunk_3s.len() as f64 / 16000.0;

            use std::cell::Cell;
            let stream_tokens = Cell::new(0usize);
            let phys_before_call = process_phys_footprint_mb().unwrap_or(0);
            let tracker = PeakMemTracker::start();
            let t0 = Instant::now();
            let result = model.transcribe_streaming(chunk_3s, &default_opts(), |_t, _w, _ts| {
                stream_tokens.set(stream_tokens.get() + 1);
            })?;
            let elapsed_ms = t0.elapsed().as_millis();
            let peak_phys = tracker.stop();
            let phys_after_call = process_phys_footprint_mb().unwrap_or(0);
            let stream_tokens = stream_tokens.get();
            let delta = phys_after_call as i64 - phys_before_call as i64;
            let rtf = elapsed_ms as f64 / 1000.0 / actual_dur;
            let word_count = result.text.split_whitespace().count();

            println!(
                "[fallback 3s] {:.2}с → \"{}\" ({} мс, RTF={:.3}, {} tokens)",
                actual_dur,
                truncate(&result.text, 80),
                elapsed_ms,
                rtf,
                stream_tokens
            );

            results.push(BenchResult {
                label: "fallback".to_string(),
                duration_sec: actual_dur,
                inference_ms: elapsed_ms,
                rtf,
                text: result.text,
                word_count,
                streaming_token_count: stream_tokens,
                peak_phys_mb: peak_phys,
                delta_phys_mb: delta,
            });
        }
    }

    let rss_final = process_rss_mb().unwrap_or(0);
    let phys_final = process_phys_footprint_mb().unwrap_or(0);
    let metal_final = metal_alloc_mb().unwrap_or(0);
    println!();
    println!("FINAL (после всех transcribe + drop):");
    println!("  RSS:             {} МБ (delta от старта: +{} МБ)", rss_final, rss_final.saturating_sub(rss_before));
    println!("  Phys footprint:  {} МБ (delta: +{} МБ) ← ЧЕСТНАЯ", phys_final, phys_final.saturating_sub(phys_before));
    println!("  IOKit/Metal:     {} МБ (delta: +{} МБ)", metal_final, metal_final.saturating_sub(metal_before));

    print_table(&results);

    // Summary metrics
    if !results.is_empty() {
        let avg_rtf: f64 = results.iter().map(|r| r.rtf).sum::<f64>() / results.len() as f64;
        let max_rtf = results.iter().map(|r| r.rtf).fold(0.0f64, f64::max);
        let peak_phys_all: u64 = results.iter().map(|r| r.peak_phys_mb).max().unwrap_or(0);
        println!();
        println!("СВОДКА:");
        println!("  Avg RTF:                  {:.3}", avg_rtf);
        println!("  Max RTF:                  {:.3} (worst case)", max_rtf);
        println!("  Peak phys footprint:      {} МБ (max across runs — ВКЛЮЧАЕТ Metal)", peak_phys_all);
        println!("  Final phys delta:         +{} МБ over baseline", phys_final.saturating_sub(phys_before));
        println!("  Load time:                {} мс", load_ms);
        println!("  Warmup:                   {} мс", warmup_ms);
        println!();
        println!("ВНИМАНИЕ: 'RSS' через ps -o rss= НЕ включает Metal GPU buffers!");
        println!("Используй 'Phys footprint' — это правда (footprint CLI / mach phys_footprint).");
    }

    Ok(())
}
