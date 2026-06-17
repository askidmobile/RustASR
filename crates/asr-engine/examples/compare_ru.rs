//! Сравнительный тест всех ASR-моделей RustASR на русском языке.
//!
//! Прогоняет Whisper / GigaAM / Parakeet / Qwen3-ASR / Nemotron на одних и тех же
//! русских аудио, печатает транскрипт + RTF + время загрузки + размер весов,
//! и (опционально) WER против эталона.
//!
//! Запуск (CPU):
//!   cargo run -p asr-engine --example compare_ru --release -- [audio1.wav ...]
//! Metal:
//!   RUSTASR_TEST_DEVICE=metal cargo run -p asr-engine --example compare_ru --release
//!
//! Модели берутся из ../../models/<dir>. Отсутствующие — пропускаются.

use std::path::{Path, PathBuf};
use std::time::Instant;

use asr_core::{AsrModel, ModelType, TranscribeOptions};
use asr_engine::AsrEngine;
use candle_core::Device;

fn models_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap().join("models")
}

fn pick_device() -> Device {
    match std::env::var("RUSTASR_TEST_DEVICE").as_deref() {
        Ok("metal") => Device::new_metal(0).unwrap_or(Device::Cpu),
        Ok("cuda") => Device::new_cuda(0).unwrap_or(Device::Cpu),
        _ => Device::Cpu,
    }
}

/// RIFF-парсер: i16 PCM → f32 (ищет data-чанк, не skip-44).
fn load_wav(path: &Path) -> Vec<f32> {
    let raw = std::fs::read(path).expect("read wav");
    let mut off = 12usize;
    while off + 8 <= raw.len() {
        let id = &raw[off..off + 4];
        let sz = u32::from_le_bytes([raw[off + 4], raw[off + 5], raw[off + 6], raw[off + 7]]) as usize;
        let body = off + 8;
        if id == b"data" {
            let end = (body + sz).min(raw.len());
            return raw[body..end].chunks_exact(2)
                .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0).collect();
        }
        off = body + sz + (sz & 1);
    }
    panic!("no data chunk in {path:?}");
}

/// Word-level WER (Левенштейн по словам). 0.0 = идеально.
fn wer(reference: &str, hyp: &str) -> f64 {
    let r: Vec<&str> = reference.split_whitespace().collect();
    let h: Vec<&str> = hyp.split_whitespace().collect();
    if r.is_empty() {
        return if h.is_empty() { 0.0 } else { 1.0 };
    }
    let (n, m) = (r.len(), h.len());
    let mut dp = vec![vec![0usize; m + 1]; n + 1];
    for i in 0..=n { dp[i][0] = i; }
    for j in 0..=m { dp[0][j] = j; }
    for i in 1..=n {
        for j in 1..=m {
            let cost = if r[i - 1] == h[j - 1] { 0 } else { 1 };
            dp[i][j] = (dp[i - 1][j] + 1).min(dp[i][j - 1] + 1).min(dp[i - 1][j - 1] + cost);
        }
    }
    dp[n][m] as f64 / n as f64
}

fn dir_size_mb(dir: &Path) -> f64 {
    // Размер ШИПУЕМОГО артефакта: .gguf приоритетно (квантованный), иначе .safetensors.
    let (mut gguf, mut st) = (0u64, 0u64);
    if let Ok(rd) = std::fs::read_dir(dir) {
        for e in rd.flatten() {
            let p = e.path();
            match p.extension().and_then(|s| s.to_str()).unwrap_or("") {
                "gguf" => if let Ok(m) = std::fs::metadata(&p) { gguf = gguf.max(m.len()); },
                "safetensors" => if let Ok(m) = std::fs::metadata(&p) { st = st.max(m.len()); },
                _ => {}
            }
        }
    }
    (if gguf > 0 { gguf } else { st }) as f64 / 1e6
}

fn main() {
    let device = pick_device();
    let root = models_root();

    let args: Vec<String> = std::env::args().skip(1).collect();
    let audios: Vec<PathBuf> = if args.is_empty() {
        let r = PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap().to_path_buf();
        vec![r.join("test_prod_3s.wav"), r.join("test_30sec.wav")]
    } else {
        args.iter().map(PathBuf::from).collect()
    };

    let models = [
        (ModelType::GigaAm, "gigaam-v3-e2e-ctc"),
        (ModelType::Whisper, "whisper-large-v3-turbo"),
        (ModelType::Parakeet, "parakeet-tdt-0.6b-v3"),
        (ModelType::Qwen3Asr, "qwen3-asr-0.6b"),
        (ModelType::Nemotron, "nemotron-3.5-asr-streaming-0.6b"),
    ];

    println!("device={device:?}\n");

    for audio in &audios {
        if !audio.exists() {
            println!("# SKIP audio {audio:?} (нет файла)\n");
            continue;
        }
        let samples = load_wav(audio);
        let dur = samples.len() as f64 / 16000.0;
        println!("================ {} ({:.1}s) ================", audio.file_name().unwrap().to_string_lossy(), dur);

        // Эталон для WER: транскрипт GigaAM (RU-нативная) как baseline-reference.
        let mut reference: Option<String> = None;
        let mut rows: Vec<(String, String, f64, u128, f64)> = Vec::new();

        for (mt, dir) in &models {
            let mdir = root.join(dir);
            if !mdir.exists() {
                println!("  [{:<10}] SKIP (нет {dir})", mt.as_str());
                continue;
            }
            let t0 = Instant::now();
            let eng = AsrEngine::load(*mt, &mdir, &device);
            let mut eng = match eng {
                Ok(e) => e,
                Err(e) => { println!("  [{:<10}] LOAD ERR: {e}", mt.as_str()); continue; }
            };
            let load_ms = t0.elapsed().as_millis();
            match eng.transcribe(&samples, &TranscribeOptions::default()) {
                Ok(r) => {
                    let size = dir_size_mb(&mdir);
                    if *mt == ModelType::GigaAm { reference = Some(r.text.clone()); }
                    rows.push((mt.display_name().to_string(), r.text, r.rtf, load_ms, size));
                }
                Err(e) => println!("  [{:<10}] INFER ERR: {e}", mt.as_str()),
            }
        }

        println!("\n  {:<26} {:>6} {:>8} {:>8}  {}", "model", "RTF", "load_ms", "size_MB", "WER%(vs GigaAM)");
        for (name, text, rtf, load_ms, size) in &rows {
            let w = reference.as_ref().map(|r| wer(r, text) * 100.0).unwrap_or(f64::NAN);
            println!("  {name:<26} {rtf:>6.3} {load_ms:>8} {size:>8.0}  {w:>6.1}");
        }
        println!();
        for (name, text, _, _, _) in &rows {
            println!("  {name:<26}: {text:?}");
        }
        println!();
    }
}
