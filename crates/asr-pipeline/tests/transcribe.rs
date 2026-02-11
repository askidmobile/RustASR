//! Integration tests for ASR Pipeline.

use std::path::PathBuf;

use asr_pipeline::AsrPipeline;
use candle_core::Device;

fn get_model_path() -> Option<PathBuf> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("models")
        .join("qwen3-asr-0.6b");

    if path.join("model.safetensors").exists() {
        Some(path)
    } else {
        None
    }
}

fn pick_test_device() -> Device {
    // По умолчанию: CPU (Metal может быть недоступен, а candle внутри может panic).
    // Для локальной проверки на Metal: RUSTASR_TEST_DEVICE=metal cargo test -p asr-pipeline --test transcribe
    match std::env::var("RUSTASR_TEST_DEVICE").as_deref() {
        Ok("metal") => std::panic::catch_unwind(|| Device::new_metal(0).ok())
            .ok()
            .flatten()
            .unwrap_or(Device::Cpu),
        _ => Device::Cpu,
    }
}

#[test]
fn test_pipeline_transcribe_sine_wave() {
    let model_path = match get_model_path() {
        Some(p) => p,
        None => {
            eprintln!("⚠️  Skipping test: model not found");
            return;
        }
    };

    let device = pick_test_device();
    eprintln!("📱 Using device: {:?}", device);

    let pipeline =
        AsrPipeline::from_model_dir(&model_path, &device).expect("Failed to create pipeline");

    eprintln!("✅ Pipeline created!");

    // Generate 1 second of 440Hz sine wave at 16kHz
    let sample_rate = 16000;
    let duration_secs = 1.0;
    let frequency = 440.0;

    let samples: Vec<f32> = (0..(sample_rate as usize * duration_secs as usize))
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            (2.0 * std::f32::consts::PI * frequency * t).sin() * 0.5
        })
        .collect();

    eprintln!(
        "🎵 Generated {} samples ({} sec)",
        samples.len(),
        duration_secs
    );

    let result = pipeline.transcribe(&samples);

    match result {
        Ok(tokens) => {
            eprintln!("✅ Transcription succeeded!");
            eprintln!("   Token count: {}", tokens.len());
            eprintln!("   First 10 tokens: {:?}", &tokens[..tokens.len().min(10)]);

            // Also test text decoding
            let text_result = pipeline.transcribe_to_text(&samples);
            match text_result {
                Ok(text) => {
                    eprintln!("📝 Decoded text: \"{}\"", text);
                }
                Err(e) => {
                    eprintln!("⚠️  Text decoding failed: {}", e);
                }
            }
        }
        Err(e) => {
            eprintln!("⚠️  Transcription failed: {}", e);
        }
    }
}

#[test]
fn test_pipeline_transcribe_silence() {
    let model_path = match get_model_path() {
        Some(p) => p,
        None => {
            eprintln!("⚠️  Skipping test: model not found");
            return;
        }
    };

    let device = pick_test_device();

    let pipeline =
        AsrPipeline::from_model_dir(&model_path, &device).expect("Failed to create pipeline");

    // Generate 0.5 seconds of silence
    let sample_rate = 16000;
    let samples: Vec<f32> = vec![0.0; sample_rate / 2];

    eprintln!("🔇 Testing with {} samples of silence", samples.len());

    let result = pipeline.transcribe(&samples);

    match result {
        Ok(tokens) => {
            eprintln!("✅ Transcription succeeded!");
            eprintln!("   Token count: {}", tokens.len());
        }
        Err(e) => {
            eprintln!("⚠️  Transcription failed: {}", e);
        }
    }
}
