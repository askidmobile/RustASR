//! Parity conformer-слоёв энкодера против эталона NeMo.
//! layer0_out, layer1_out (как [B,T,D]) и полный encoded_pre_prompt ([B,D,T]).
//!
//! `cargo test -p model-nemotron --test encoder_layers_parity -- --nocapture`

use std::path::PathBuf;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use model_nemotron::{NemotronConfig, NemotronEncoder};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap().to_path_buf()
}

fn stats(a: &Tensor, b: &Tensor) -> (f64, f64) {
    let a = a.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let b = b.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let (mut s, mut m) = (0.0f64, 0.0f64);
    for (x, y) in a.iter().zip(b.iter()) {
        let d = (*x as f64 - *y as f64).abs();
        s += d;
        if d > m {
            m = d;
        }
    }
    (s / a.len() as f64, m)
}

#[test]
fn conformer_layers_match_nemo() {
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
    let enc = NemotronEncoder::load(&config.encoder, vb.pp("encoder")).unwrap();

    let refs = candle_core::safetensors::load(&parity, &device).unwrap();
    let mel = refs.get("mel").unwrap();

    // layer0/layer1 hook-дампы NeMo ненадёжны в cache-aware режиме (информативно).
    let l0 = enc.forward_through(mel, 1).unwrap();
    let (m0, x0) = stats(&l0, refs.get("layer0_out").unwrap());
    println!("[info] layer0: mean_abs={m0:.5} max_abs={x0:.5}");
    let l1 = enc.forward_through(mel, 2).unwrap();
    let (m1, x1) = stats(&l1, refs.get("layer1_out").unwrap());
    println!("[info] layer1: mean_abs={m1:.5} max_abs={x1:.5}");

    let full = enc.forward(mel).unwrap(); // [B,D,T]
    let refr = refs.get("encoded_pre_prompt").unwrap();
    let (mf, xf) = stats(&full, refr);

    // Где макс ошибка? (последний кадр маскируется NeMo → безвреден)
    let (_b, d, t) = full.dims3().unwrap();
    let fa = full.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let fb = refr.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let (mut argmax, mut mx) = (0usize, 0.0f64);
    for i in 0..fa.len() {
        let dd = (fa[i] as f64 - fb[i] as f64).abs();
        if dd > mx { mx = dd; argmax = i; }
    }
    println!("encoder parity: mean_abs={mf:.5} max_abs={xf:.5} @ (d={}, t={})", argmax / t, argmax % t);

    // Interior: исключаем последний кадр (t-1), который NeMo маскирует.
    let interior = full.narrow(2, 0, t - 1).unwrap();
    let refi = refr.narrow(2, 0, t - 1).unwrap();
    let (mi, xi) = stats(&interior, &refi);
    println!("encoder interior (без last frame): mean_abs={mi:.5} max_abs={xi:.5}  (d={d}, t={t})");

    assert!(mi < 0.01, "encoder interior mean_abs={mi}");
    assert!(xi < 0.2, "encoder interior max_abs={xi}");
}
