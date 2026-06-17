//! NemotronModel — точка сборки пайплайна (Ф3, в разработке).
//!
//! Сейчас: загрузка config.json + валидация артефактов. Инференс — TODO (Ф3):
//! mel → encoder(causal) → prompt_kernel → joint → RNN-T greedy → detok,
//! с parity-проверкой против `tmp/parity/nemotron_*.npy`.

use std::path::{Path, PathBuf};

use candle_core::Device;
use tracing::info;

use asr_core::{AsrError, AsrResult};

use crate::config::NemotronConfig;

/// Модель Nemotron 3.5 ASR (FastConformer-CacheAware-RNNT + Lang prompt).
pub struct NemotronModel {
    pub config: NemotronConfig,
    #[allow(dead_code)]
    device: Device,
    #[allow(dead_code)]
    model_dir: PathBuf,
}

impl NemotronModel {
    /// Загрузить модель из директории (config.json + model.safetensors + vocab.json).
    ///
    /// Ф3: на текущем этапе грузится только конфиг и проверяется наличие весов;
    /// сборка энкодера/декодера/joint — следующий шаг.
    pub fn load(model_dir: impl AsRef<Path>, device: &Device) -> AsrResult<Self> {
        let model_dir = model_dir.as_ref().to_path_buf();
        let config_path = model_dir.join("config.json");
        if !config_path.exists() {
            return Err(AsrError::Config(format!(
                "config.json не найден в {model_dir:?}"
            )));
        }
        let config = NemotronConfig::from_json_file(&config_path)
            .map_err(|e| AsrError::Config(format!("Ошибка парсинга config.json: {e}")))?;

        for required in ["model.safetensors", "vocab.json"] {
            if !model_dir.join(required).exists() {
                return Err(AsrError::Model(format!(
                    "Nemotron: не найден {required} в {model_dir:?}"
                )));
            }
        }

        info!(
            "Nemotron config загружен: {} (encoder {}×{}d, vocab {}, prompt ru-RU={:?})",
            config.model_name,
            config.encoder.n_layers,
            config.encoder.d_model,
            config.decoder.vocab_size,
            config.prompt.lang_idx("ru-RU"),
        );

        Ok(Self {
            config,
            device: device.clone(),
            model_dir,
        })
    }

    /// Транскрипция — TODO Ф3 (mel → encoder → prompt_kernel → RNN-T greedy → detok).
    pub fn transcribe_stub(&self) -> AsrResult<String> {
        Err(AsrError::Model(
            "Nemotron инференс ещё не реализован (Ф3 в разработке)".into(),
        ))
    }
}
