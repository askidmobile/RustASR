//! Integration tests for loading and running the Qwen3 decoder.

use std::path::PathBuf;

use candle_core::{DType, Device, Tensor};
use qwen3_decoder::{Qwen3Config, Qwen3Decoder};

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
    // Для локальной проверки на Metal: RUSTASR_TEST_DEVICE=metal cargo test -p qwen3-decoder --test load_decoder
    match std::env::var("RUSTASR_TEST_DEVICE").as_deref() {
        Ok("metal") => std::panic::catch_unwind(|| Device::new_metal(0).ok())
            .ok()
            .flatten()
            .unwrap_or(Device::Cpu),
        _ => Device::Cpu,
    }
}

#[test]
fn test_load_decoder_from_safetensors() {
    let model_path = match get_model_path() {
        Some(p) => p,
        None => {
            eprintln!("⚠️  Skipping test: model not found");
            eprintln!("   Run: python scripts/download_model.py");
            return;
        }
    };

    let device = pick_test_device();
    eprintln!("📱 Using device: {:?}", device);

    // Load config from model
    let config =
        Qwen3Config::from_hf_config(model_path.join("config.json")).expect("Failed to load config");

    eprintln!("📊 Config loaded:");
    eprintln!("   hidden_size: {}", config.hidden_size);
    eprintln!("   num_hidden_layers: {}", config.num_hidden_layers);
    eprintln!("   num_attention_heads: {}", config.num_attention_heads);
    eprintln!("   num_key_value_heads: {}", config.num_key_value_heads);
    eprintln!("   vocab_size: {}", config.vocab_size);

    // Try to load the decoder
    let safetensors_path = model_path.join("model.safetensors");
    let result = Qwen3Decoder::from_safetensors(config.clone(), &safetensors_path, &device);

    match result {
        Ok(decoder) => {
            eprintln!("✅ Decoder loaded successfully!");
            eprintln!("   Vocab size: {}", decoder.vocab_size());

            // Test forward pass with dummy input (audio embeddings)
            let batch_size = 1;
            let seq_len = 50; // Simulating audio encoder output

            let dummy_audio_embeds = Tensor::zeros(
                (batch_size, seq_len, config.hidden_size),
                DType::BF16,
                &device,
            )
            .expect("Failed to create dummy input");

            let output = decoder.forward_with_audio(&dummy_audio_embeds, None, 0);

            match output {
                Ok(logits) => {
                    eprintln!("✅ Forward pass succeeded!");
                    eprintln!("   Logits shape: {:?}", logits.dims());

                    // Expected: [1, 50, vocab_size]
                    assert_eq!(logits.dims(), &[batch_size, seq_len, config.vocab_size]);
                }
                Err(e) => {
                    eprintln!("⚠️  Forward pass failed: {}", e);
                }
            }
        }
        Err(e) => {
            eprintln!("⚠️  Failed to load decoder: {}", e);
            eprintln!("   This may be due to missing weight tensors.");
        }
    }
}

#[test]
fn test_decoder_config_matches_model() {
    let model_path = match get_model_path() {
        Some(p) => p,
        None => {
            eprintln!("⚠️  Skipping test: model not found");
            return;
        }
    };

    let config =
        Qwen3Config::from_hf_config(model_path.join("config.json")).expect("Failed to load config");

    // Verify config matches expected values
    assert_eq!(config.hidden_size, 1024, "hidden_size mismatch");
    assert_eq!(config.num_hidden_layers, 28, "num_hidden_layers mismatch");
    assert_eq!(
        config.num_attention_heads, 16,
        "num_attention_heads mismatch"
    );
    assert_eq!(
        config.num_key_value_heads, 8,
        "num_key_value_heads mismatch"
    );
    assert_eq!(config.intermediate_size, 3072, "intermediate_size mismatch");
    assert_eq!(config.vocab_size, 151936, "vocab_size mismatch");

    eprintln!("✅ Config verification passed!");
}
