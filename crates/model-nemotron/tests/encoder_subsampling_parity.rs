//! Parity субдискретизации энкодера против эталона NeMo (pre_encode_out).
//!
//! Изолирует subsampling: на вход подаётся ЭТАЛОННЫЙ mel (NeMo), сравнивается
//! выход pre_encode. Требует локальных артефактов, иначе skip.
//!
//! Запуск: `cargo test -p model-nemotron --test encoder_subsampling_parity -- --nocapture`

use std::path::PathBuf;

use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use model_nemotron::{NemotronConfig, NemotronEncoder};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

#[test]
fn subsampling_matches_nemo_reference() {
    let root = root();
    let model_dir = root.join("models/nemotron-3.5-asr-streaming-0.6b");
    let parity = root.join("tmp/parity/nemotron_parity.safetensors");
    let weights = model_dir.join("model.safetensors");

    if !weights.exists() || !parity.exists() {
        eprintln!("SKIP: нет локальных артефактов (модель/parity)");
        return;
    }

    let device = Device::Cpu;
    let config = NemotronConfig::from_json_file(&model_dir.join("config.json")).unwrap();

    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[weights], DType::F32, &device).unwrap()
    };
    let encoder = NemotronEncoder::load(&config.encoder, vb.pp("encoder")).unwrap();

    let refs = candle_core::safetensors::load(&parity, &device).unwrap();
    let mel_ref = refs.get("mel").unwrap(); // [1,128,301]
    let pre_ref = refs.get("pre_encode_out").unwrap(); // [1,39,1024]

    let out = encoder.forward_subsampling(mel_ref).unwrap(); // [1,39,1024]
    println!("subsampling rust shape: {:?}, ref: {:?}", out.dims(), pre_ref.dims());
    assert_eq!(out.dims(), pre_ref.dims(), "формы pre_encode должны совпадать");

    let a = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let b = pre_ref.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let n = a.len() as f64;
    let (mut sum, mut max) = (0.0f64, 0.0f64);
    for (x, y) in a.iter().zip(b.iter()) {
        let d = (*x as f64 - *y as f64).abs();
        sum += d;
        if d > max {
            max = d;
        }
    }
    let mean = sum / n;
    println!("subsampling parity: mean_abs={mean:.5} max_abs={max:.5}");
    assert!(mean < 0.01, "mean_abs слишком большой: {mean}");
    assert!(max < 0.2, "max_abs слишком большой: {max}");
}
