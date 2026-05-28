//! Bench: Parakeet Q8 GGUF vs safetensors на одинаковом prod audio.
//!
//! cargo run --release --example bench_gguf -p model-parakeet --features candle-core/metal

use std::cell::Cell;
use std::path::PathBuf;
use std::time::Instant;

use asr_core::{AsrModel, TranscribeOptions};
use candle_core::{Device, IndexOp, D};
use model_parakeet::ParakeetModelGguf;

const GGUF_PATH: &str = "/tmp/parakeet-gguf/parakeet-q8_0.gguf";
const PROD_AUDIO_PATHS: &[&str] = &[
    "/Users/askid/Library/Application Support/com.yttri.app/users/askid.mobile_at_gmail.com/recordings/584ce801-f009-46ea-a4dc-aec7fabc4429/full.mp3",
];

fn process_phys_footprint_mb() -> Option<u64> {
    let pid = std::process::id().to_string();
    let output = std::process::Command::new("footprint")
        .args(["-p", &pid])
        .output()
        .ok()?;
    let s = String::from_utf8_lossy(&output.stdout);
    for line in s.lines() {
        if let Some(idx) = line.find("Footprint:") {
            let tail = &line[idx + "Footprint:".len()..];
            let parts: Vec<&str> = tail.trim().split_whitespace().collect();
            if parts.len() >= 2 {
                let num: f64 = parts[0].parse().ok()?;
                let mb = match parts[1] {
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

fn load_wav_16k(path: &str) -> Vec<f32> {
    let data = std::fs::read(path).expect("read wav");
    let pcm = &data[44..];
    pcm.chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0)
        .collect()
}

fn extract_chunk_from_prod_mp3(out_path: &str, start_sec: u64, duration_sec: u64) -> Option<String> {
    let src = PROD_AUDIO_PATHS
        .iter()
        .find(|p| std::path::Path::new(p).exists())?;
    let status = std::process::Command::new("ffmpeg")
        .args([
            "-hide_banner", "-loglevel", "error", "-y",
            "-i", src,
            "-ss", &start_sec.to_string(),
            "-t", &duration_sec.to_string(),
            "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
            out_path,
        ])
        .status()
        .ok()?;
    if !status.success() { return None; }
    Some(out_path.to_string())
}

/// Encode + TDT decode используя ParakeetModelGguf компоненты напрямую.
/// (Альтернативно — обернуть в AsrModel trait, но для bench пока инлайним).
fn transcribe_gguf(
    model: &ParakeetModelGguf,
    samples: &[f32],
) -> Result<(String, usize, usize), Box<dyn std::error::Error>> {
    use candle_core::Tensor;

    // 1. Mel spectrogram
    let mel = model.mel_extractor.extract(samples, &model.device)?;
    // 2. Encoder
    let enc_out = model.encoder.forward(&mel)?;
    let enc_out = enc_out.squeeze(0)?; // [T, d_model]

    // 3. TDT decode (manual — без callback пока)
    let t_total = enc_out.dim(0)?;
    let enc_h_all = model.joint.project_encoder_all(&enc_out)?;

    let mut tokens: Vec<u32> = Vec::new();
    let mut state = model.prediction_net.initial_state(&model.device)?;
    let mut last_token: u32 = model.config.decoder.blank_idx as u32;
    let mut time_idx: usize = 0;
    let mut pred_dirty = true;
    let mut pred_out_cached: Option<Tensor> = None;
    let max_iters = t_total * 4;
    let mut iter = 0;

    while time_idx < t_total && iter < max_iters {
        iter += 1;
        if pred_dirty {
            let (po, sn) = model.prediction_net.step(last_token, &state)?;
            pred_out_cached = Some(po);
            state = sn;
            pred_dirty = false;
        }
        let pred_out = pred_out_cached.as_ref().unwrap();
        let enc_h_frame = enc_h_all.i(time_idx)?;
        let (tl, dl) = model.joint.forward_with_cached_enc(&enc_h_frame, pred_out)?;
        let token_argmax = tl.argmax_keepdim(D::Minus1)?;
        let dur_argmax = dl.argmax_keepdim(D::Minus1)?;
        let combined = Tensor::cat(&[&token_argmax, &dur_argmax], 0)?;
        let v: Vec<u32> = combined.flatten_all()?.to_vec1()?;
        let k = v[0];
        let dur_idx = v[1] as usize;
        let skip = model.config.tdt.durations.get(dur_idx).copied().unwrap_or(1);
        if k as usize == model.config.decoder.blank_idx {
            time_idx += skip.max(1);
        } else {
            tokens.push(k);
            last_token = k;
            pred_dirty = true;
            time_idx += skip.max(1);
        }
    }

    let text = model.decode_tokens(&tokens);
    let words = text.split_whitespace().count();
    Ok((text, tokens.len(), words))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info,model_parakeet=info")),
        )
        .try_init();

    println!("=== Parakeet Q8 GGUF — Metal bench ===");
    println!();

    let phys_before = process_phys_footprint_mb().unwrap_or(0);
    println!("Phys footprint до load: {} МБ", phys_before);

    let device = Device::new_metal(0).map_err(|e| format!("Metal init: {e}"))?;

    let load_t0 = Instant::now();
    let model = ParakeetModelGguf::load(GGUF_PATH, &device)?;
    let load_ms = load_t0.elapsed().as_millis();

    let phys_after_load = process_phys_footprint_mb().unwrap_or(0);
    println!(
        "Phys footprint после load: {} МБ (delta: +{} МБ, load: {} мс)",
        phys_after_load,
        phys_after_load.saturating_sub(phys_before),
        load_ms
    );

    // Тесты
    let chunks: Vec<(&str, u64, u64)> = vec![
        ("prod 3s @ 60s", 60, 3),
        ("prod 5s @ 60s", 60, 5),
        ("prod 10s @ 60s", 60, 10),
        ("prod 30s @ 30s", 30, 30),
    ];

    println!();
    println!("--- Бенч на prod audio ---");

    for (label, start_sec, dur_sec) in chunks {
        let tmp = format!("/tmp/parakeet_q8_bench_{}_{}s.wav", start_sec, dur_sec);
        let path = match extract_chunk_from_prod_mp3(&tmp, start_sec, dur_sec) {
            Some(p) => p,
            None => {
                eprintln!("SKIP {}: ffmpeg failed", label);
                continue;
            }
        };
        let samples = load_wav_16k(&path);
        let actual_dur = samples.len() as f64 / 16000.0;

        let phys_pre = process_phys_footprint_mb().unwrap_or(0);
        let t0 = Instant::now();
        let (text, token_count, word_count) = transcribe_gguf(&model, &samples)?;
        let elapsed = t0.elapsed().as_millis();
        let phys_post = process_phys_footprint_mb().unwrap_or(0);
        let rtf = elapsed as f64 / 1000.0 / actual_dur;
        let preview: String = text.chars().take(80).collect();
        println!(
            "[{}] {:.2}с → \"{}\" ({} мс, RTF={:.3}, {} tokens, {} words, phys: {}→{})",
            label, actual_dur, preview, elapsed, rtf, token_count, word_count, phys_pre, phys_post
        );

        let _ = std::fs::remove_file(&path);
    }

    let phys_final = process_phys_footprint_mb().unwrap_or(0);
    println!();
    println!("Phys footprint final: {} МБ (delta от start: +{} МБ)",
        phys_final, phys_final.saturating_sub(phys_before));

    Ok(())
}

#[allow(dead_code)]
fn _suppress_warns() {
    let _ = PathBuf::new();
    let _: Option<Cell<usize>> = None;
}
