//! Квантование Nemotron: model.safetensors (F32) → model-q8_0.gguf.
//!
//! Стратегия (как run_quantize в asr-cli, «как Qwen ASR» — Q8_0 максимально сохраняет качество):
//! - rank-2 `.weight` и НЕ norm → Q8_0 (encoder FFN/attn, pre_encode.out, prompt_kernel,
//!   joint, decoder embed — основная масса 609M энкодера)
//! - всё прочее (нормы, bias'ы rank-1, conv rank-3/4, fb/window, LSTM weight_ih/hh,
//!   pos_bias) → F16
//!
//! Запуск:
//!   cargo run -p model-nemotron --example quantize_nemotron --release -- \
//!     models/nemotron-3.5-asr-streaming-0.6b/model.safetensors \
//!     models/nemotron-3.5-asr-streaming-0.6b/model-q8_0.gguf

use std::collections::BTreeMap;

use candle_core::quantized::{gguf_file, GgmlDType, QTensor};
use candle_core::{DType, Device};

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: quantize_nemotron <input.safetensors> <output.gguf>");
        std::process::exit(1);
    }
    let input = &args[1];
    let output = &args[2];
    // По умолчанию Q8 ТОЛЬКО энкодер (609M); joint/decoder/prompt в F16 — выше точность
    // argmax (RNN-T тонкие margin'ы), +21МБ. 3-й арг "all" → квантовать всё.
    let encoder_only = args.get(3).map(|s| s != "all").unwrap_or(true);
    let device = Device::Cpu;

    let tensors = candle_core::safetensors::load(input, &device)?;
    // BTreeMap — детерминированный порядок записи.
    let tensors: BTreeMap<_, _> = tensors.into_iter().collect();

    let mut out: Vec<(String, QTensor)> = Vec::new();
    let (mut nq, mut nk) = (0usize, 0usize);
    let (mut bytes_q, mut bytes_f) = (0usize, 0usize);

    for (name, t) in tensors {
        let is_norm = name.contains("norm") || name.contains("layernorm");
        let in_scope = !encoder_only || name.starts_with("encoder.");
        let q8 = t.rank() == 2 && name.ends_with(".weight") && !is_norm && in_scope;
        if q8 {
            let qt = QTensor::quantize(&t, GgmlDType::Q8_0)?;
            bytes_q += t.elem_count(); // ~1 байт/элемент Q8_0 (+ overhead)
            out.push((name, qt));
            nq += 1;
        } else {
            let t16 = if t.dtype() == DType::F16 { t } else { t.to_dtype(DType::F16)? };
            bytes_f += t16.elem_count() * 2;
            let qt = QTensor::quantize(&t16, GgmlDType::F16)?;
            out.push((name, qt));
            nk += 1;
        }
    }

    let mut f = std::fs::File::create(output)?;
    let refs: Vec<(&str, &QTensor)> = out.iter().map(|(n, t)| (n.as_str(), t)).collect();
    gguf_file::write(&mut f, &[], &refs)?;

    let sz = std::fs::metadata(output)?.len();
    println!(
        "quantized: {nq} Q8_0 (~{:.0}MB) + {nk} F16 (~{:.0}MB) → {output} ({:.1} MB)",
        bytes_q as f64 / 1e6,
        bytes_f as f64 / 1e6,
        sz as f64 / 1e6
    );
    Ok(())
}
