//! Проверка десериализации реального config.json (skip, если модель не скачана).

use std::path::PathBuf;

use model_nemotron::NemotronConfig;

#[test]
fn parses_real_config_json() {
    // crates/model-nemotron → workspace root → models/...
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../models/nemotron-3.5-asr-streaming-0.6b/config.json");
    if !path.exists() {
        eprintln!("skip: {path:?} отсутствует (модель не сконвертирована)");
        return;
    }
    let cfg = NemotronConfig::from_json_file(&path).expect("config.json должен парситься");

    assert_eq!(cfg.encoder.n_layers, 24);
    assert_eq!(cfg.encoder.d_model, 1024);
    assert_eq!(cfg.preprocessor.features, 128);
    assert_eq!(cfg.preprocessor.win_length(), 400);
    assert_eq!(cfg.preprocessor.hop_length(), 160);
    assert_eq!(cfg.decoder.blank_idx, 13087);
    assert_eq!(cfg.joint.output_dim, 13088);
    assert_eq!(cfg.prompt.in_features, 1152);
    assert_eq!(cfg.prompt.lang_idx("ru-RU"), Some(11));
}
