//! End-to-end транскрипция Nemotron — финальный арбитр parity (текст vs golden NeMo).
//!
//! `cargo test -p model-nemotron --test transcribe_e2e -- --nocapture`

use std::path::PathBuf;

use candle_core::Device;
use model_nemotron::NemotronModel;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap().to_path_buf()
}

fn load_wav_i16(path: &PathBuf) -> Vec<f32> {
    let raw = std::fs::read(path).expect("read wav");
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
    panic!("no data chunk");
}

#[test]
fn transcribes_russian() {
    let root = root();
    let model_dir = root.join("models/nemotron-3.5-asr-streaming-0.6b");
    if !model_dir.join("model.safetensors").exists() {
        eprintln!("SKIP: модель не сконвертирована");
        return;
    }
    let device = Device::Cpu;
    let model = NemotronModel::load(&model_dir, &device).unwrap();

    // Изоляция joint+greedy: подаём ЭТАЛОННЫЙ fused (NeMo prompt_kernel_out).
    let parity = root.join("tmp/parity/nemotron_parity.safetensors");
    if parity.exists() {
        let refs = candle_core::safetensors::load(&parity, &device).unwrap();
        let fused_ref = refs.get("prompt_kernel_out").unwrap(); // [1,39,1024]
        let toks = model.greedy_decode(fused_ref).unwrap();
        let text = model.decode_text(&toks);
        println!("[isolation] greedy на эталонном fused → {} токенов: {text:?}", toks.len());
    }

    // golden NeMo (ru-RU): test_prod_3s.wav → "Примерно в двадцать третьем году в двадцать четыре"
    // Паритет проверяем на GREEDY (бит-в-бит к NeMo greedy). Beam — отдельно (smoke).
    let wav = root.join("test_prod_3s.wav");
    let golden = "Примерно в двадцать третьем году в двадцать четыре";
    let samples = load_wav_i16(&wav);

    unsafe { std::env::set_var("NEMOTRON_BEAM", "1") }; // форсируем greedy
    let text = model.transcribe(&samples).unwrap();
    println!("GREEDY: {text:?}");
    println!("GOLD  : {golden:?}");
    assert_eq!(text.trim(), golden.trim(), "greedy должен совпасть с эталоном NeMo");

    // beam (дефолт) — smoke: непустой и содержит ключевые слова.
    unsafe { std::env::remove_var("NEMOTRON_BEAM") };
    let beam = model.transcribe(&samples).unwrap();
    println!("BEAM  : {beam:?}");
    assert!(beam.contains("двадцать") && beam.contains("Примерно"), "beam должен дать связный RU");
}
