//! Inspect Parakeet Q8 GGUF: load metadata + tensors, print summary.
//!
//! cargo run --release --example inspect_gguf -p model-parakeet --features candle-core/metal -- /tmp/parakeet-gguf/parakeet-q8_0.gguf

use candle_core::Device;
use model_parakeet::ParakeetGguf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/tmp/parakeet-gguf/parakeet-q8_0.gguf".to_string());

    let device = Device::new_metal(0).unwrap_or(Device::Cpu);
    println!("Device: {:?}", device);
    println!("Loading: {}", path);

    let t0 = std::time::Instant::now();
    let gguf = ParakeetGguf::from_file(&path, &device)?;
    let load_ms = t0.elapsed().as_millis();

    println!();
    println!("=== Loaded in {} мс ===", load_ms);
    println!();
    println!("Config:");
    println!("  model: {}", gguf.config.model_name);
    println!("  sample_rate: {}", gguf.config.sample_rate);
    println!("  n_mels: {}", gguf.config.preprocessor.features);
    println!("  d_model: {}", gguf.config.encoder.d_model);
    println!("  n_layers: {}", gguf.config.encoder.n_layers);
    println!("  n_heads: {}", gguf.config.encoder.n_heads);
    println!("  vocab_size: {}", gguf.config.decoder.vocab_size);
    println!("  blank_idx: {}", gguf.config.decoder.blank_idx);
    println!("  TDT durations: {:?}", gguf.config.tdt.durations);
    println!();
    println!("Vocab: {} tokens", gguf.vocab.len());
    println!("  first 5: {:?}", &gguf.vocab[..5.min(gguf.vocab.len())]);
    println!();
    println!(
        "Mel filterbank: shape={:?}, dtype={:?}",
        gguf.mel_filterbank.shape(),
        gguf.mel_filterbank.dtype()
    );
    println!(
        "Mel window: shape={:?}, dtype={:?}",
        gguf.mel_window.shape(),
        gguf.mel_window.dtype()
    );
    println!();

    let mut quant = 0;
    let mut reg_f16 = 0;
    let mut reg_f32 = 0;
    for (name, t) in &gguf.tensors {
        let _ = name;
        if t.is_quantized() {
            quant += 1;
        } else if let Some(rt) = t.as_regular() {
            match rt.dtype() {
                candle_core::DType::F16 => reg_f16 += 1,
                candle_core::DType::F32 => reg_f32 += 1,
                _ => {}
            }
        }
    }
    println!("Tensor count by type:");
    println!("  Q8_0 (quantized): {}", quant);
    println!("  F16:              {}", reg_f16);
    println!("  F32:              {}", reg_f32);
    println!("  TOTAL:            {}", gguf.tensors.len());

    // Sample tensor checks
    println!();
    println!("Sample tensors:");
    for name in &[
        "encoder.layers.0.ff1.linear1.weight",
        "encoder.layers.0.attn.q.weight",
        "encoder.layers.0.conv.dw.weight",
        "encoder.layers.0.conv.bn.weight",
        "decoder.lstm.0.w_ih",
        "decoder.embed.weight",
        "joint.enc.weight",
        "joint.out.weight",
        "encoder.pre.conv.0.weight",
        "encoder.pre.out.weight",
    ] {
        match gguf.get(name) {
            Ok(t) => {
                let info = if let Some(qt) = t.as_quantized() {
                    format!("Q8_0 shape={:?}", qt.shape())
                } else if let Some(rt) = t.as_regular() {
                    format!("{:?} shape={:?}", rt.dtype(), rt.shape())
                } else {
                    "?".to_string()
                };
                println!("  {}: {}", name, info);
            }
            Err(e) => println!("  {}: ❌ {}", name, e),
        }
    }

    println!();
    println!("✅ GGUF loader работает. Готов к интеграции в encoder/decoder/joint.");

    Ok(())
}
