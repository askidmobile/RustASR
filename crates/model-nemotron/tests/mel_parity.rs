//! Parity mel-экстрактора против эталона NeMo (tmp/parity/nemotron_mel.npy → safetensors).
//!
//! Требует локальных артефактов (модель + parity-дамп), иначе skip:
//!   models/nemotron-3.5-asr-streaming-0.6b/{model.safetensors,config.json}
//!   tmp/parity/nemotron_parity.safetensors  (scripts/parity_npy_to_safetensors.py)
//!   test_prod_3s.wav (корень workspace)
//!
//! Запуск: `cargo test -p model-nemotron --test mel_parity -- --nocapture`

use std::path::PathBuf;

use candle_core::Device;
use model_nemotron::{NemotronConfig, NemotronMelExtractor};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

/// Минимальный RIFF-парсер: находит чанк `data` (заголовок не всегда 44 байта!)
/// и читает i16 PCM → f32/32768 (как soundfile dtype=float32).
fn load_wav_i16(path: &PathBuf) -> Vec<f32> {
    let raw = std::fs::read(path).expect("read wav");
    assert_eq!(&raw[0..4], b"RIFF");
    assert_eq!(&raw[8..12], b"WAVE");
    let mut off = 12usize;
    while off + 8 <= raw.len() {
        let id = &raw[off..off + 4];
        let sz = u32::from_le_bytes([raw[off + 4], raw[off + 5], raw[off + 6], raw[off + 7]]) as usize;
        let body = off + 8;
        if id == b"data" {
            let end = (body + sz).min(raw.len());
            return raw[body..end]
                .chunks_exact(2)
                .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0)
                .collect();
        }
        off = body + sz + (sz & 1);
    }
    panic!("data chunk не найден в {path:?}");
}

#[test]
fn mel_matches_nemo_reference() {
    let root = root();
    let model_dir = root.join("models/nemotron-3.5-asr-streaming-0.6b");
    let parity = root.join("tmp/parity/nemotron_parity.safetensors");
    let wav = root.join("test_prod_3s.wav");

    if !model_dir.join("model.safetensors").exists() || !parity.exists() || !wav.exists() {
        eprintln!("SKIP mel_parity: нет локальных артефактов (модель/parity/wav)");
        return;
    }

    let device = Device::Cpu;
    let config = NemotronConfig::from_json_file(&model_dir.join("config.json")).unwrap();

    // fb + window из весов модели
    let weights =
        candle_core::safetensors::load(model_dir.join("model.safetensors"), &device).unwrap();
    let fb = weights.get("preprocessor.featurizer.fb").expect("fb");
    let window = weights.get("preprocessor.featurizer.window").expect("window");

    let extractor = NemotronMelExtractor::from_tensors(&config, fb, window).unwrap();

    let samples = load_wav_i16(&wav);
    let mel = extractor.extract(&samples, &device).unwrap(); // [1,128,T]

    // эталон
    let refs = candle_core::safetensors::load(&parity, &device).unwrap();
    let mel_ref = refs.get("mel").expect("mel ref"); // [1,128,301]

    println!("mel rust   shape: {:?}", mel.dims());
    println!("mel ref    shape: {:?}", mel_ref.dims());
    assert_eq!(mel.dims(), mel_ref.dims(), "формы mel должны совпадать");

    let a = mel.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let b = mel_ref.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let n = a.len() as f64;
    let mut sum_abs = 0.0f64;
    let mut max_abs = 0.0f64;
    let mut argmax = 0usize;
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        let d = (*x as f64 - *y as f64).abs();
        sum_abs += d;
        if d > max_abs {
            max_abs = d;
            argmax = i;
        }
    }
    let mean_abs = sum_abs / n;
    let t = mel_ref.dims()[2];
    println!(
        "mel parity: mean_abs={mean_abs:.5} max_abs={max_abs:.5} @ (mel={}, frame={})",
        argmax / t,
        argmax % t
    );

    assert!(mean_abs < 0.05, "mean_abs слишком большой: {mean_abs}");
    assert!(max_abs < 1.0, "max_abs слишком большой: {max_abs}");
}
