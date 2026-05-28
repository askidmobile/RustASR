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

/// macOS process RSS via mach API.
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
}

fn print_table(rows: &[BenchResult]) {
    println!();
    println!("┌─────────────────────────────────┬──────────┬──────────┬────────┬──────────┬───────┐");
    println!("│ label                           │ audio    │ infer    │ RTF    │ stream   │ words │");
    println!("│                                 │ (sec)    │ (ms)     │        │ tokens   │       │");
    println!("├─────────────────────────────────┼──────────┼──────────┼────────┼──────────┼───────┤");
    for r in rows {
        println!(
            "│ {:<31} │ {:>8.2} │ {:>8} │ {:>6.3} │ {:>8} │ {:>5} │",
            truncate(&r.label, 31),
            r.duration_sec,
            r.inference_ms,
            r.rtf,
            r.streaming_token_count,
            r.word_count
        );
    }
    println!("└─────────────────────────────────┴──────────┴──────────┴────────┴──────────┴───────┘");
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

    let rss_before_load = process_rss_mb().unwrap_or(0);
    println!("RSS до загрузки модели: {} МБ", rss_before_load);

    let device = Device::new_metal(0).map_err(|e| format!("Metal init: {e}"))?;
    println!("Device: Metal GPU (M-series)");

    let load_t0 = Instant::now();
    let mut model = ParakeetModel::load(MODEL_PATH, &device)?;
    let load_ms = load_t0.elapsed().as_millis();

    let rss_after_load = process_rss_mb().unwrap_or(0);
    println!(
        "Загрузка: {} мс, RSS: {} МБ → {} МБ (delta: +{} МБ)",
        load_ms,
        rss_before_load,
        rss_after_load,
        rss_after_load.saturating_sub(rss_before_load)
    );

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

        // Замер streaming: считаем сколько токенов эмитится через callback.
        use std::cell::Cell;
        let stream_tokens = Cell::new(0usize);

        let t0 = Instant::now();
        let result = model
            .transcribe_streaming(&samples, &default_opts(), |_text, _is_wb, _ts_ms| {
                stream_tokens.set(stream_tokens.get() + 1);
            })?;
        let elapsed_ms = t0.elapsed().as_millis();
        let stream_tokens = stream_tokens.get();

        let rtf = elapsed_ms as f64 / 1000.0 / actual_dur;
        let word_count = result.text.split_whitespace().count();

        println!(
            "[{}] {:.2}с → \"{}\" ({} мс, RTF={:.3}, {} stream tokens, {} words)",
            label,
            actual_dur,
            truncate(&result.text, 80),
            elapsed_ms,
            rtf,
            stream_tokens,
            word_count
        );

        results.push(BenchResult {
            label: label.to_string(),
            duration_sec: actual_dur,
            inference_ms: elapsed_ms,
            rtf,
            text: result.text,
            word_count,
            streaming_token_count: stream_tokens,
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
            let t0 = Instant::now();
            let result = model.transcribe_streaming(chunk_3s, &default_opts(), |_t, _w, _ts| {
                stream_tokens.set(stream_tokens.get() + 1);
            })?;
            let elapsed_ms = t0.elapsed().as_millis();
            let stream_tokens = stream_tokens.get();
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
                label: "fallback-test_real_audio 3s".to_string(),
                duration_sec: actual_dur,
                inference_ms: elapsed_ms,
                rtf,
                text: result.text,
                word_count,
                streaming_token_count: stream_tokens,
            });
        }
    }

    let rss_final = process_rss_mb().unwrap_or(0);
    println!();
    println!("RSS финальный: {} МБ (delta от старта: +{} МБ)", rss_final, rss_final.saturating_sub(rss_before_load));

    print_table(&results);

    // Summary metrics
    if !results.is_empty() {
        let avg_rtf: f64 = results.iter().map(|r| r.rtf).sum::<f64>() / results.len() as f64;
        let max_rtf = results.iter().map(|r| r.rtf).fold(0.0f64, f64::max);
        println!();
        println!("СВОДКА:");
        println!("  Avg RTF:     {:.3}", avg_rtf);
        println!("  Max RTF:     {:.3} (worst case)", max_rtf);
        println!("  Memory:      +{} МБ over baseline", rss_final.saturating_sub(rss_before_load));
        println!("  Load time:   {} мс", load_ms);
        println!("  Warmup:      {} мс", warmup_ms);
    }

    Ok(())
}
