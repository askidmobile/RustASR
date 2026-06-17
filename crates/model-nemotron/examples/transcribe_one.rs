//! Транскрипция одного wav через NemotronModel (для диагностики качества).
//! NEMOTRON_FORCE_F32=1 → safetensors; NEMOTRON_GGUF=<file> → конкретный gguf.
//!
//!   cargo run -p model-nemotron --example transcribe_one -- <model_dir> <wav>

use std::path::{Path, PathBuf};

use candle_core::Device;
use model_nemotron::NemotronModel;

fn load_wav(path: &Path) -> Vec<f32> {
    let raw = std::fs::read(path).expect("read wav");
    let mut off = 12usize;
    while off + 8 <= raw.len() {
        let id = &raw[off..off + 4];
        let sz = u32::from_le_bytes([raw[off + 4], raw[off + 5], raw[off + 6], raw[off + 7]]) as usize;
        let body = off + 8;
        if id == b"data" {
            let end = (body + sz).min(raw.len());
            return raw[body..end].chunks_exact(2)
                .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0).collect();
        }
        off = body + sz + (sz & 1);
    }
    panic!("no data chunk");
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let model_dir = PathBuf::from(&args[0]);
    let wav = PathBuf::from(&args[1]);
    let device = Device::Cpu;
    let model = NemotronModel::load(&model_dir, &device).unwrap();
    let samples = load_wav(&wav);
    let text = model.transcribe(&samples).unwrap();
    let mode = if std::env::var("NEMOTRON_FORCE_F32").is_ok() {
        "F32".to_string()
    } else {
        format!("GGUF:{}", std::env::var("NEMOTRON_GGUF").unwrap_or_else(|_| "model-q8_0.gguf".into()))
    };
    println!("[{mode}] {text:?}");
}
