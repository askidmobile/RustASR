//! Интеграционные тесты Parakeet TDT 0.6B v3.
//!
//! Запуск:
//! ```bash
//! # CPU (по умолчанию):
//! cargo test -p model-parakeet --test transcribe -- --nocapture
//!
//! # Metal GPU:
//! RUSTASR_TEST_DEVICE=metal cargo test -p model-parakeet --test transcribe -- --nocapture
//! ```

use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use asr_core::{AsrModel, TranscribeOptions};
use candle_core::Device;
use model_parakeet::ParakeetModel;

fn get_model_path() -> Option<PathBuf> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("models")
        .join("parakeet-tdt-0.6b-v3");

    if path.join("model.safetensors").exists() && path.join("vocab.json").exists() {
        Some(path)
    } else {
        None
    }
}

fn pick_test_device() -> Device {
    match std::env::var("RUSTASR_TEST_DEVICE").as_deref() {
        Ok("metal") => std::panic::catch_unwind(|| Device::new_metal(0).ok())
            .ok()
            .flatten()
            .unwrap_or(Device::Cpu),
        _ => Device::Cpu,
    }
}

fn load_wav(path: &str) -> Vec<f32> {
    let data = std::fs::read(path).expect("Failed to read WAV file");
    // Skip WAV header (44 bytes), read i16 PCM samples
    let pcm = &data[44..];
    pcm.chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0)
        .collect()
}

fn default_options() -> TranscribeOptions {
    TranscribeOptions {
        language: None,
        max_tokens: None,
        timestamps: true,
        temperature: 0.0,
    }
}

// ============================================================================
// Тест 1: Загрузка модели
// ============================================================================

#[test]
fn test_parakeet_load() {
    let model_path = match get_model_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: Parakeet model not found");
            return;
        }
    };

    let device = pick_test_device();
    eprintln!("Device: {:?}", device);

    let start = Instant::now();
    let model = ParakeetModel::load(&model_path, &device);
    let elapsed = start.elapsed();

    match model {
        Ok(m) => {
            eprintln!(
                "Parakeet loaded in {:.2}s: name={}, type={:?}",
                elapsed.as_secs_f64(),
                m.name(),
                m.model_type(),
            );
            let info = m.model_info();
            eprintln!(
                "  params={:?}, weights={:?}, backend={}",
                info.parameters, info.weights_size_bytes, info.backend,
            );
            eprintln!("  languages: {:?}", m.supported_languages());
        }
        Err(e) => {
            panic!("Parakeet load FAILED: {}", e);
        }
    }
}

// ============================================================================
// Тест 2: Транскрипция 3с аудио
// ============================================================================

#[test]
fn test_parakeet_transcribe_3s() {
    let model_path = match get_model_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: Parakeet model not found");
            return;
        }
    };

    let wav_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("test_real_audio.wav");

    if !wav_path.exists() {
        eprintln!("SKIP: test_real_audio.wav not found");
        return;
    }

    let device = pick_test_device();
    let mut model = ParakeetModel::load(&model_path, &device).expect("Load failed");

    let all_samples = load_wav(wav_path.to_str().unwrap());
    let chunk_5s = 5 * 16000;
    let samples = &all_samples[..chunk_5s.min(all_samples.len())];
    let duration = samples.len() as f64 / 16000.0;
    eprintln!("Audio: {:.1}s, {} samples (from test_real_audio.wav)", duration, samples.len());

    let start = Instant::now();
    let result = model.transcribe(&samples, &default_options());
    let elapsed = start.elapsed();

    match result {
        Ok(r) => {
            let rtf = elapsed.as_secs_f64() / duration;
            eprintln!(
                "Transcription: \"{}\" ({:.0}ms, RTF={:.3})",
                r.text,
                elapsed.as_millis(),
                rtf,
            );
            eprintln!("  segments: {}", r.segments.len());
            for seg in &r.segments {
                eprintln!("    [{:.2}s-{:.2}s] \"{}\"", seg.start, seg.end, seg.text);
            }

            assert!(!r.text.is_empty(), "Transcription is empty");
            assert!(rtf < 5.0, "RTF too high: {:.2}", rtf);
        }
        Err(e) => {
            panic!("Transcription FAILED: {}", e);
        }
    }
}

// ============================================================================
// Тест 3: Транскрипция 30с аудио
// ============================================================================

#[test]
fn test_parakeet_transcribe_30s() {
    let model_path = match get_model_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: Parakeet model not found");
            return;
        }
    };

    let wav_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("test_30sec.wav");

    if !wav_path.exists() {
        eprintln!("SKIP: test_30sec.wav not found");
        return;
    }

    let device = pick_test_device();
    let mut model = ParakeetModel::load(&model_path, &device).expect("Load failed");

    let samples = load_wav(wav_path.to_str().unwrap());
    let duration = samples.len() as f64 / 16000.0;
    eprintln!("Audio: {:.1}s, {} samples", duration, samples.len());

    let start = Instant::now();
    let result = model.transcribe(&samples, &default_options());
    let elapsed = start.elapsed();

    match result {
        Ok(r) => {
            let rtf = elapsed.as_secs_f64() / duration;
            eprintln!(
                "Transcription: \"{}\" ({:.0}ms, RTF={:.3})",
                &r.text[..r.text.char_indices().nth(60).map(|(i,_)| i).unwrap_or(r.text.len())],
                elapsed.as_millis(),
                rtf,
            );
            eprintln!("  segments: {}", r.segments.len());

            assert!(!r.text.is_empty(), "Transcription is empty");
        }
        Err(e) => {
            panic!("Transcription 30s FAILED: {}", e);
        }
    }
}

// ============================================================================
// Тест 4: Streaming callback
// ============================================================================

#[test]
fn test_parakeet_streaming() {
    let model_path = match get_model_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: Parakeet model not found");
            return;
        }
    };

    let wav_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("test_real_audio.wav");

    if !wav_path.exists() {
        eprintln!("SKIP: test_real_audio.wav not found");
        return;
    }

    let device = pick_test_device();
    let mut model = ParakeetModel::load(&model_path, &device).expect("Load failed");

    let all_samples = load_wav(wav_path.to_str().unwrap());
    let chunk_5s = 5 * 16000;
    let samples = all_samples[..chunk_5s.min(all_samples.len())].to_vec();
    let token_count = AtomicUsize::new(0);
    let word_count = AtomicUsize::new(0);

    let result = model.transcribe_streaming(&samples, &default_options(), |text, is_wb, ts_ms| {
        token_count.fetch_add(1, Ordering::Relaxed);
        if is_wb {
            word_count.fetch_add(1, Ordering::Relaxed);
        }
        eprint!("{}", text);
        let _ = ts_ms;
    });

    eprintln!(); // newline after streaming output

    match result {
        Ok(r) => {
            let tokens = token_count.load(Ordering::Relaxed);
            let words = word_count.load(Ordering::Relaxed);
            eprintln!(
                "Streaming: {} tokens, {} words, final=\"{}\"",
                tokens, words, r.text,
            );
            eprintln!("  segments: {}", r.segments.len());

            assert!(tokens > 0, "No streaming tokens emitted");
            assert!(!r.text.is_empty(), "Final text is empty");
        }
        Err(e) => {
            panic!("Streaming FAILED: {}", e);
        }
    }
}

// ============================================================================
// Тест 5: TDT decoder с NeMo encoder output (bypass encoder)
// ============================================================================

#[test]
fn test_parakeet_tdt_with_nemo_encoder() {
    let model_path = match get_model_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: Parakeet model not found");
            return;
        }
    };

    let nemo_enc_path = "/tmp/nemo_encoder_out.npy";
    if !std::path::Path::new(nemo_enc_path).exists() {
        eprintln!("SKIP: /tmp/nemo_encoder_out.npy not found (run parakeet_debug_reference.py first)");
        return;
    }

    let device = pick_test_device();
    let model = ParakeetModel::load(&model_path, &device).expect("Load failed");

    // Load NeMo encoder output [63, 1024] as raw f32 from .npy
    let npy_data = std::fs::read(nemo_enc_path).expect("Failed to read npy");
    // numpy .npy: magic(6) + version(2) + header_len(2 or 4) + header + padding + data
    // Find data start: scan for \n that ends the header
    let mut data_start = 10; // skip magic + version + header_len
    while data_start < npy_data.len() && npy_data[data_start - 1] != b'\n' {
        data_start += 1;
    }
    let dims = [63usize, 1024]; // known from NeMo output
    let float_data: Vec<f32> = npy_data[data_start..]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    eprintln!("Loaded {} floats (expected {})", float_data.len(), dims[0] * dims[1]);

    let enc_tensor = candle_core::Tensor::from_vec(float_data, (dims[0], dims[1]), &device).unwrap();
    eprintln!("Encoder tensor: {:?}", enc_tensor.shape());

    // Run TDT decoder
    let result = model.tdt_decoder.decode(
        &enc_tensor,
        &model.prediction_net,
        &model.joint,
    ).expect("TDT decode failed");

    let text = model.decode_tokens(&result.tokens);
    eprintln!("TDT result from NeMo encoder: \"{}\" ({} tokens)", text, result.tokens.len());

    if text.is_empty() {
        eprintln!("WARNING: still empty — decoder or joint might have issues too");
    } else {
        eprintln!("SUCCESS: TDT decoder works with NeMo encoder output!");
    }
}

// ============================================================================
// Тест 6: Тишина — модель не должна падать
// ============================================================================

#[test]
fn test_parakeet_silence() {
    let model_path = match get_model_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: Parakeet model not found");
            return;
        }
    };

    let device = pick_test_device();
    let mut model = ParakeetModel::load(&model_path, &device).expect("Load failed");

    let samples: Vec<f32> = vec![0.0; 16000]; // 1s silence
    let result = model.transcribe(&samples, &default_options());

    match result {
        Ok(r) => {
            eprintln!("Silence result: \"{}\" (segments: {})", r.text, r.segments.len());
        }
        Err(e) => {
            panic!("Silence transcription FAILED: {}", e);
        }
    }
}
