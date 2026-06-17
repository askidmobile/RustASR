//! Parity prompt_kernel против эталона NeMo: на вход prompt_kernel_in [1,39,1152],
//! сравниваем с prompt_kernel_out [1,39,1024].
//!
//! `cargo test -p model-nemotron --test prompt_parity -- --nocapture`

use std::path::PathBuf;

use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use model_nemotron::{NemotronConfig, PromptKernel};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap().to_path_buf()
}

#[test]
fn prompt_kernel_matches_nemo() {
    let root = root();
    let model_dir = root.join("models/nemotron-3.5-asr-streaming-0.6b");
    let parity = root.join("tmp/parity/nemotron_parity.safetensors");
    let weights = model_dir.join("model.safetensors");
    if !weights.exists() || !parity.exists() {
        eprintln!("SKIP: нет локальных артефактов");
        return;
    }
    let device = Device::Cpu;
    let config = NemotronConfig::from_json_file(&model_dir.join("config.json")).unwrap();
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights], DType::F32, &device).unwrap() };
    let pk = PromptKernel::load(&config.prompt, vb.pp("prompt_kernel")).unwrap();

    let refs = candle_core::safetensors::load(&parity, &device).unwrap();
    let fin = refs.get("prompt_kernel_in").unwrap();
    let fout = refs.get("prompt_kernel_out").unwrap();

    let out = pk.forward_fused(fin).unwrap();
    assert_eq!(out.dims(), fout.dims());
    let a = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let b = fout.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let (mut s, mut m) = (0.0f64, 0.0f64);
    for (x, y) in a.iter().zip(b.iter()) {
        let d = (*x as f64 - *y as f64).abs();
        s += d;
        if d > m { m = d; }
    }
    let mean = s / a.len() as f64;
    println!("prompt_kernel parity: mean_abs={mean:.6} max_abs={m:.6}");
    assert!(mean < 1e-3, "mean_abs={mean}");
    assert!(m < 1e-2, "max_abs={m}");
}
