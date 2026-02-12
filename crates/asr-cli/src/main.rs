//!
//! CLI для распознавания речи (Qwen3-ASR, Whisper и другие модели).

mod speaker;
mod vad;

use anyhow::Result;
use clap::{Parser, Subcommand, ValueEnum};
use std::path::{Path, PathBuf};
use std::time::Instant;

use audio::{Resampler, load_wav, loader::to_mono};

/// Тип ASR-модели для CLI.
#[derive(ValueEnum, Debug, Clone, Copy, PartialEq, Eq)]
enum ModelTypeArg {
    /// Qwen3-ASR (по умолчанию, обратная совместимость)
    Qwen3,
    /// OpenAI Whisper (Large v3 Turbo и другие)
    Whisper,
    /// GigaAM v3 E2E CTC
    Gigaam,
    /// Parakeet TDT
    Parakeet,
}

#[derive(Parser)]
#[command(name = "rustasr")]
#[command(author, version, about = "RustASR: Qwen3-based Speech Recognition", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

impl ModelTypeArg {
    fn to_model_type(self) -> asr_core::ModelType {
        match self {
            ModelTypeArg::Qwen3 => asr_core::ModelType::Qwen3Asr,
            ModelTypeArg::Whisper => asr_core::ModelType::Whisper,
            ModelTypeArg::Gigaam => asr_core::ModelType::GigaAm,
            ModelTypeArg::Parakeet => asr_core::ModelType::Parakeet,
        }
    }
}

#[derive(ValueEnum, Debug, Clone, Copy)]
enum DecoderWeightsArg {
    Auto,
    Safetensors,
    Gguf,
}

impl From<DecoderWeightsArg> for asr_pipeline::DecoderWeights {
    fn from(v: DecoderWeightsArg) -> Self {
        match v {
            DecoderWeightsArg::Auto => asr_pipeline::DecoderWeights::Auto,
            DecoderWeightsArg::Safetensors => asr_pipeline::DecoderWeights::Safetensors,
            DecoderWeightsArg::Gguf => asr_pipeline::DecoderWeights::Gguf,
        }
    }
}

#[derive(ValueEnum, Debug, Clone, Copy)]
enum SpeakerModeArg {
    /// Автовыбор: mono => cluster, stereo => channel
    Auto,
    /// Говорящий = канал (левый/правый)
    Channel,
    /// Кластеризация говорящих по акустическим признакам (для mono/миксов)
    Cluster,
}

#[derive(Subcommand)]
enum Commands {
    /// Transcribe an audio file to text
    Transcribe {
        /// Path to the model directory
        #[arg(long)]
        model: PathBuf,

        /// Тип модели: qwen3 (по умолчанию) или whisper
        #[arg(long, value_enum, default_value = "qwen3")]
        model_type: ModelTypeArg,

        /// Path to the audio file (WAV format)
        #[arg(long)]
        audio: PathBuf,

        /// Device to use (cpu, metal, cuda)
        #[arg(long, default_value = "cpu")]
        device: String,

        /// Maximum number of tokens to generate (по умолчанию подбирается автоматически)
        #[arg(long)]
        max_tokens: Option<usize>,

        /// Форсировать язык вывода (как в qwen-asr: добавляет `language X<asr_text>` в промпт)
        #[arg(long)]
        language: Option<String>,

        /// Печатать сырой вывод модели (включая `language ...<asr_text>`)
        #[arg(long, default_value_t = false)]
        raw: bool,

        /// Откуда загружать веса декодера: auto|safetensors|gguf
        #[arg(long, value_enum, default_value = "auto")]
        decoder_weights: DecoderWeightsArg,

        /// Явно указать GGUF-файл для декодера (путь или имя файла внутри --model).
        ///
        /// Удобно для сравнений (например, `model-q6k.gguf`) без переименований.
        #[arg(long)]
        decoder_gguf: Option<PathBuf>,

        /// Сохранить итоговый текст распознавания в файл (UTF-8).
        ///
        /// В файл пишется именно "очищенный" текст (`result.text`), без служебных токенов.
        #[arg(long)]
        out_text: Option<PathBuf>,
    },

    /// Run a simple test to verify the setup
    Test {
        /// Device to use (cpu, metal)
        #[arg(long, default_value = "cpu")]
        device: String,
    },

    /// Диаризация (VAD) + транскрибация по сегментам.
    ///
    /// Полезно для длинных записей:
    /// - `--speaker-mode channel`: левый/правый канал = разные источники (mic/system).
    /// - `--speaker-mode cluster`: mono или смешанные дорожки, выделение говорящих кластеризацией.
    Diarize {
        /// Path to the model directory
        #[arg(long)]
        model: PathBuf,

        /// Тип модели: qwen3 (по умолчанию) или whisper/gigaam/parakeet
        #[arg(long, value_enum, default_value = "qwen3")]
        model_type: ModelTypeArg,

        /// Path to the audio file (WAV format)
        #[arg(long)]
        audio: PathBuf,

        /// Device to use (cpu, metal, cuda)
        #[arg(long, default_value = "cpu")]
        device: String,

        /// Откуда загружать веса декодера: auto|safetensors|gguf
        #[arg(long, value_enum, default_value = "auto")]
        decoder_weights: DecoderWeightsArg,

        /// Явно указать GGUF-файл для декодера (путь или имя файла внутри --model).
        #[arg(long)]
        decoder_gguf: Option<PathBuf>,

        /// Максимум токенов на один сегмент (если не задано, подбирается по длительности).
        #[arg(long)]
        max_tokens: Option<usize>,

        /// Форсировать язык распознавания (суффикс `language X<asr_text>`).
        ///
        /// Пример: `--language Russian`.
        #[arg(long)]
        language: Option<String>,

        /// Форсировать язык для конкретного "говорящего" (имеет приоритет над `--language`).
        ///
        /// Можно задавать несколько раз:
        /// `--speaker-language mic=Russian --speaker-language system=English`
        #[arg(long, value_name = "SPEAKER=LANG")]
        speaker_language: Vec<String>,

        /// VAD mode (0=quality, 1=low-bitrate, 2=aggressive, 3=very-aggressive)
        #[arg(long, default_value_t = 2)]
        vad_mode: u8,

        /// VAD frame length in ms (10/20/30)
        #[arg(long, default_value_t = 30)]
        vad_frame_ms: usize,

        /// Minimum speech duration to start a segment (ms)
        #[arg(long, default_value_t = 300)]
        vad_min_speech_ms: usize,

        /// Minimum silence duration to close a segment (ms)
        #[arg(long, default_value_t = 200)]
        vad_min_silence_ms: usize,

        /// Padding added to each segment (ms)
        #[arg(long, default_value_t = 150)]
        vad_pad_ms: usize,

        /// Maximum segment duration (seconds)
        #[arg(long, default_value_t = 30)]
        vad_max_segment_s: usize,

        /// Режим определения "говорящих": auto|channel|cluster.
        ///
        /// - `channel`: спикер = канал (левый/правый).
        /// - `cluster`: спикеры выделяются кластеризацией (полезно для mono и смешанных дорожек).
        /// - `auto`: mono => cluster, stereo => channel.
        #[arg(long, value_enum, default_value = "auto")]
        speaker_mode: SpeakerModeArg,

        /// Количество говорящих для speaker-mode=cluster.
        ///
        /// Если не задано, используется 2 (частый кейс "разговор").
        #[arg(long)]
        num_speakers: Option<usize>,

        /// Имена говорящих для cluster-режима (через запятую).
        ///
        /// Пример: `--speaker-names "Я,Собеседник"` (для 2-х спикеров).
        #[arg(long, value_delimiter = ',')]
        speaker_names: Vec<String>,

        /// Имя "говорящего" для левого канала.
        #[arg(long, default_value = "mic")]
        left_speaker: String,

        /// Имя "говорящего" для правого канала.
        #[arg(long, default_value = "system")]
        right_speaker: String,

        /// Кластеризовать правый канал на N говорящих (полезно, если в `system` несколько собеседников).
        ///
        /// Работает только при stereo и effective speaker-mode=`channel`.
        /// Если не задано, правый канал считается одним "говорящим".
        #[arg(long)]
        right_num_speakers: Option<usize>,

        /// Имена говорящих для правого канала (через запятую).
        ///
        /// Пример: `--right-speaker-names "alice,bob"` при `--right-num-speakers 2`.
        #[arg(long, value_delimiter = ',')]
        right_speaker_names: Vec<String>,

        /// Куда писать результаты (report.md, segments.json, speaker texts).
        #[arg(long)]
        out_dir: Option<PathBuf>,
    },

    /// Convert safetensors weights to GGUF (quantized)
    Quantize {
        /// Input model directory or model.safetensors path
        #[arg(long, short = 'i')]
        input: PathBuf,

        /// Output GGUF file path
        #[arg(long, short = 'o')]
        output: PathBuf,

        /// Quantization type: q8_0, q6k, q4_0
        #[arg(long, default_value = "q8_0")]
        qtype: String,

        /// Only include tensors with this prefix (default: thinker.model.)
        #[arg(long, default_value = "thinker.model.")]
        scope: String,

        /// Quantize embeddings (embed_tokens.weight). Default: false.
        #[arg(long, default_value_t = false)]
        quantize_embeddings: bool,
    },

    /// Работа с локальными моделями (поиск/проверка файлов)
    Models {
        #[command(subcommand)]
        command: ModelsCommands,
    },
}

#[derive(Subcommand)]
enum ModelsCommands {
    /// Показать модели в директории (по умолчанию ./models)
    List {
        /// Директория, в которой лежат подпапки с моделями
        #[arg(long, default_value = "models")]
        root: PathBuf,
    },

    /// Проверить конкретную директорию модели и вывести, какие файлы найдены
    Check {
        /// Путь к директории модели
        #[arg(long)]
        model: PathBuf,
    },
}

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Transcribe {
            model,
            model_type,
            audio,
            device,
            max_tokens,
            language,
            raw,
            decoder_weights,
            decoder_gguf,
            out_text,
        } => {
            println!("🎤 RustASR - Speech Recognition");
            println!("================================");
            println!("Model: {}", model.display());
            println!("Model type: {:?}", model_type);
            println!("Audio file: {}", audio.display());
            println!("Device: {}", device);
            println!();

            let start = Instant::now();

            println!("📂 Loading audio file...");
            let audio_buffer = load_wav(&audio)?;
            println!(
                "   Sample rate: {} Hz, Duration: {:.2}s",
                audio_buffer.sample_rate,
                audio_buffer.duration()
            );

            let mono_buffer = to_mono(&audio_buffer);

            println!("🔄 Resampling to 16kHz...");
            let resampler = Resampler::new(16000);
            let resampled = resampler.resample(&mono_buffer)?;
            println!("   Samples: {}", resampled.samples.len());

            let device = create_device(&device)?;

            match model_type {
                ModelTypeArg::Whisper => {
                    run_transcribe_whisper(
                        &model, &resampled.samples, &device, language.as_deref(),
                        max_tokens, out_text, start, &decoder_weights,
                    )?;
                }
                ModelTypeArg::Gigaam => {
                    run_transcribe_gigaam(
                        &model, &resampled.samples, &device, out_text, start,
                    )?;
                }
                ModelTypeArg::Parakeet => {
                    run_transcribe_parakeet(
                        &model, &resampled.samples, &device, out_text, start,
                    )?;
                }
                ModelTypeArg::Qwen3 => {
                    run_transcribe_qwen3(
                        &model, &resampled.samples, &device, language.as_deref(),
                        max_tokens, raw, decoder_weights, decoder_gguf, out_text, start,
                    )?;
                }
            }

            Ok(())
        }

        Commands::Test { device } => {
            println!("🧪 RustASR - Setup Test");
            println!("=======================");

            let device = match device.as_str() {
                "metal" => {
                    println!("Testing Metal device...");
                    create_device("metal")?
                }
                _ => {
                    println!("Testing CPU device...");
                    create_device("cpu")?
                }
            };

            let a = candle_core::Tensor::randn(0f32, 1f32, (2, 3), &device)?;
            let b = candle_core::Tensor::randn(0f32, 1f32, (3, 4), &device)?;
            let c = a.matmul(&b)?;
            println!("✅ Test passed. Output shape: {:?}", c.dims());
            Ok(())
        }

        Commands::Diarize {
            model,
            model_type,
            audio,
            device,
            decoder_weights,
            decoder_gguf,
            max_tokens,
            language,
            speaker_language,
            vad_mode,
            vad_frame_ms,
            vad_min_speech_ms,
            vad_min_silence_ms,
            vad_pad_ms,
            vad_max_segment_s,
            speaker_mode,
            num_speakers,
            speaker_names,
            left_speaker,
            right_speaker,
            right_num_speakers,
            right_speaker_names,
            out_dir,
        } => {
            run_diarize(
                model,
                model_type,
                audio,
                &device,
                decoder_weights,
                decoder_gguf,
                max_tokens,
                language,
                speaker_language,
                vad_mode,
                vad_frame_ms,
                vad_min_speech_ms,
                vad_min_silence_ms,
                vad_pad_ms,
                vad_max_segment_s,
                speaker_mode,
                num_speakers,
                speaker_names,
                &left_speaker,
                &right_speaker,
                right_num_speakers,
                right_speaker_names,
                out_dir,
            )?;
            Ok(())
        }

        Commands::Quantize {
            input,
            output,
            qtype,
            scope,
            quantize_embeddings,
        } => {
            run_quantize(input, output, &qtype, &scope, quantize_embeddings)?;
            Ok(())
        }

        Commands::Models { command } => match command {
            ModelsCommands::List { root } => run_models_list(root),
            ModelsCommands::Check { model } => run_models_check(model),
        },
    }
}

// ---------------------------------------------------------------------------
// Whisper транскрибация (через asr-engine)
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn run_transcribe_whisper(
    model_dir: &Path,
    samples: &[f32],
    device: &candle_core::Device,
    language: Option<&str>,
    max_tokens: Option<usize>,
    out_text: Option<PathBuf>,
    start: Instant,
    decoder_weights: &DecoderWeightsArg,
) -> Result<()> {
    let quantized = matches!(decoder_weights, DecoderWeightsArg::Gguf);
    println!("🧠 Loading Whisper model{}...", if quantized { " (GGUF quantized)" } else { "" });
    let mut engine = if quantized {
        asr_engine::AsrEngine::load_quantized(
            asr_core::ModelType::Whisper,
            model_dir,
            device,
        )?
    } else {
        asr_engine::AsrEngine::load(
            asr_core::ModelType::Whisper,
            model_dir,
            device,
        )?
    };
    let info = engine.model_info();
    println!("   Model: {}", engine.name());
    println!(
        "   Parameters: {}",
        info.parameters
            .map(|p| format!("~{}M", p / 1_000_000))
            .unwrap_or_else(|| "unknown".into())
    );
    println!("   Quantization: {}", info.quantization);
    println!("   Model loaded in {:.2}s", start.elapsed().as_secs_f32());

    println!();
    println!("🎯 Transcribing...");
    let transcribe_start = Instant::now();

    let mut options = asr_core::TranscribeOptions::default();
    if let Some(lang) = language {
        options = options.with_language(lang);
    }
    if let Some(mt) = max_tokens {
        options = options.with_max_tokens(mt);
    }

    let result = engine.transcribe(samples, &options)?;
    let transcribe_time = transcribe_start.elapsed();

    println!();
    println!("═══════════════════════════════════════════");
    println!("📝 Transcription:");
    println!();
    if let Some(ref lang) = result.language {
        println!("   Language: {}", lang);
    }
    println!("   {}", result.text);

    if let Some(path) = out_text {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        std::fs::write(&path, result.text.as_bytes())?;
        println!();
        println!("💾 Saved transcription text to: {}", path.display());
    }

    if !result.segments.is_empty() {
        println!();
        println!("📊 Segments:");
        for seg in &result.segments {
            let conf = seg.confidence.map(|c| format!(" ({:.0}%)", c * 100.0)).unwrap_or_default();
            println!(
                "   [{:.1}s - {:.1}s]{} {}",
                seg.start, seg.end, conf, seg.text
            );
        }
    }

    println!();
    println!("═══════════════════════════════════════════");
    println!();
    println!(
        "⏱️  Transcription time: {:.2}s (RTF: {:.3})",
        transcribe_time.as_secs_f32(),
        result.rtf,
    );
    println!("⏱️  Total time: {:.2}s", start.elapsed().as_secs_f32());

    Ok(())
}

// ---------------------------------------------------------------------------
// GigaAM v3 E2E CTC транскрибация
// ---------------------------------------------------------------------------

fn run_transcribe_gigaam(
    model_dir: &Path,
    samples: &[f32],
    device: &candle_core::Device,
    out_text: Option<PathBuf>,
    start: Instant,
) -> Result<()> {
    println!("🧠 Загрузка модели GigaAM...");
    let mut engine = asr_engine::AsrEngine::load(
        asr_core::ModelType::GigaAm,
        model_dir,
        device,
    )?;
    let info = engine.model_info();
    println!("   Модель: {}", engine.name());
    println!(
        "   Параметры: {}",
        info.parameters
            .map(|p| format!("~{}M", p / 1_000_000))
            .unwrap_or_else(|| "unknown".into())
    );
    println!("   Квантизация: {}", info.quantization);
    println!(
        "   Модель загружена за {:.2}s",
        start.elapsed().as_secs_f32()
    );

    println!();
    println!("🎯 Транскрибация...");
    let transcribe_start = Instant::now();

    // GigaAM — только русский, без дополнительных опций
    let options = asr_core::TranscribeOptions::default().with_language("ru");
    let result = engine.transcribe(samples, &options)?;
    let transcribe_time = transcribe_start.elapsed();

    println!();
    println!("═══════════════════════════════════════════");
    println!("📝 Транскрипция:");
    println!();
    println!("   {}", result.text);

    if let Some(path) = out_text {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        std::fs::write(&path, result.text.as_bytes())?;
        println!();
        println!("💾 Текст сохранён в: {}", path.display());
    }

    if !result.segments.is_empty() {
        println!();
        println!("📊 Сегменты:");
        for seg in &result.segments {
            let conf = seg
                .confidence
                .map(|c| format!(" ({:.0}%)", c * 100.0))
                .unwrap_or_default();
            println!(
                "   [{:.1}s - {:.1}s]{} {}",
                seg.start, seg.end, conf, seg.text
            );
        }
    }

    println!();
    println!("═══════════════════════════════════════════");
    println!();
    println!(
        "⏱️  Время транскрибации: {:.2}s (RTF: {:.3})",
        transcribe_time.as_secs_f32(),
        result.rtf,
    );
    println!(
        "⏱️  Общее время: {:.2}s",
        start.elapsed().as_secs_f32()
    );

    Ok(())
}

// ---------------------------------------------------------------------------
// Parakeet TDT транскрибация
// ---------------------------------------------------------------------------

fn run_transcribe_parakeet(
    model_dir: &Path,
    samples: &[f32],
    device: &candle_core::Device,
    out_text: Option<PathBuf>,
    start: Instant,
) -> Result<()> {
    println!("🧠 Загрузка модели Parakeet TDT...");
    let mut engine = asr_engine::AsrEngine::load(
        asr_core::ModelType::Parakeet,
        model_dir,
        device,
    )?;
    let info = engine.model_info();
    println!("   Модель: {}", engine.name());
    println!(
        "   Параметры: {}",
        info.parameters
            .map(|p| format!("~{}M", p / 1_000_000))
            .unwrap_or_else(|| "unknown".into())
    );
    println!("   Квантизация: {}", info.quantization);
    println!(
        "   Модель загружена за {:.2}s",
        start.elapsed().as_secs_f32()
    );

    println!();
    println!("🎯 Транскрибация...");
    let transcribe_start = Instant::now();

    let options = asr_core::TranscribeOptions::default();
    let result = engine.transcribe(samples, &options)?;
    let transcribe_time = transcribe_start.elapsed();

    println!();
    println!("═══════════════════════════════════════════");
    println!("📝 Транскрипция:");
    println!();
    println!("   {}", result.text);

    if let Some(path) = out_text {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        std::fs::write(&path, result.text.as_bytes())?;
        println!();
        println!("💾 Текст сохранён в: {}", path.display());
    }

    if !result.segments.is_empty() {
        println!();
        println!("📊 Сегменты:");
        for seg in &result.segments {
            let conf = seg
                .confidence
                .map(|c| format!(" ({:.0}%)", c * 100.0))
                .unwrap_or_default();
            println!(
                "   [{:.1}s - {:.1}s]{} {}",
                seg.start, seg.end, conf, seg.text
            );
        }
    }

    println!();
    println!("═══════════════════════════════════════════");
    println!();
    println!(
        "⏱️  Время транскрибации: {:.2}s (RTF: {:.3})",
        transcribe_time.as_secs_f32(),
        result.rtf,
    );
    println!(
        "⏱️  Общее время: {:.2}s",
        start.elapsed().as_secs_f32()
    );

    Ok(())
}

// ---------------------------------------------------------------------------
// Qwen3-ASR транскрибация (через asr-pipeline, обратная совместимость)
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn run_transcribe_qwen3(
    model_dir: &Path,
    samples: &[f32],
    device: &candle_core::Device,
    language: Option<&str>,
    max_tokens: Option<usize>,
    raw: bool,
    decoder_weights: DecoderWeightsArg,
    decoder_gguf: Option<PathBuf>,
    out_text: Option<PathBuf>,
    start: Instant,
) -> Result<()> {
    println!("🧠 Loading Qwen3-ASR model...");
    let pipeline = asr_pipeline::AsrPipeline::from_model_dir_with_decoder_weights_and_gguf(
        model_dir,
        device,
        decoder_weights.into(),
        decoder_gguf,
    )?;
    println!("   Vocab size: {}", pipeline.vocab_size());
    println!("   Model loaded in {:.2}s", start.elapsed().as_secs_f32());

    println!();
    println!("🎯 Transcribing...");
    let transcribe_start = Instant::now();

    let duration_sec = samples.len() as f32 / 16000.0;
    let max_tokens = max_tokens.unwrap_or_else(|| estimate_max_tokens(duration_sec));

    let result = pipeline.transcribe_to_result_with_max_tokens_and_language(
        samples,
        max_tokens,
        language,
    )?;

    let transcribe_time = transcribe_start.elapsed();

    println!();
    println!("═══════════════════════════════════════════");
    println!("📝 Transcription:");
    println!();
    if raw {
        println!("   {}", result.raw);
    } else {
        if !result.language.is_empty() {
            println!("   Language: {}", result.language);
        }
        println!("   {}", result.text);
    }

    if let Some(path) = out_text {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        std::fs::write(&path, result.text.as_bytes())?;
        println!();
        println!("💾 Saved transcription text to: {}", path.display());
    }

    match result.stop_reason {
        asr_pipeline::StopReason::MaxTokens => {
            println!();
            println!(
                "   [!] Достигнут лимит max-tokens={}. Возможно, текст обрезан.",
                max_tokens
            );
            println!(
                "       Попробуйте увеличить: --max-tokens {}",
                max_tokens.saturating_mul(2)
            );
        }
        asr_pipeline::StopReason::Repetition => {
            println!();
            println!(
                "   [!] Обнаружено зацикливание. Остановлено эвристикой.",
            );
            println!(
                "       Рекомендация: используйте VAD (`rustasr diarize`) \
                 или повысите качество квантования (Q6K/Q8)."
            );
        }
        _ => {}
    }
    println!();
    println!("═══════════════════════════════════════════");
    println!();
    println!(
        "⏱️  Transcription time: {:.2}s",
        transcribe_time.as_secs_f32()
    );
    println!("⏱️  Total time: {:.2}s", start.elapsed().as_secs_f32());

    Ok(())
}

fn estimate_max_tokens(duration_sec: f32) -> usize {
    // Эвристика: запас токенов пропорционален длительности.
    // Для русского/английского безопасно брать 12-16 BPE токенов/сек + небольшой оверхед.
    // Слишком большой лимит не мешает: модель обычно останавливается по EOS.
    const TOKENS_PER_SEC: f32 = 16.0;
    const OVERHEAD: usize = 64;
    const MIN: usize = 256;
    const MAX: usize = 4096;

    let est = (duration_sec * TOKENS_PER_SEC).ceil() as usize + OVERHEAD;
    est.clamp(MIN, MAX)
}

#[derive(Debug)]
struct ModelInspection {
    model_dir: PathBuf,
    has_config: bool,
    has_vocab: bool,
    has_merges: bool,
    has_mel_filters: bool,
    gguf_files: Vec<PathBuf>,
    preferred_gguf: Option<PathBuf>,
    safetensors_files: Option<Vec<PathBuf>>,
    safetensors_error: Option<String>,
}

impl ModelInspection {
    fn is_ready_for_transcribe(&self) -> bool {
        self.has_config && self.has_vocab && self.has_merges && self.safetensors_files.is_some()
    }

    fn is_ready_for_quantize(&self) -> bool {
        self.safetensors_files.is_some()
    }
}

fn inspect_model_dir(model_dir: &Path) -> ModelInspection {
    let has_config = model_dir.join("config.json").exists();
    let has_vocab = model_dir.join("vocab.json").exists();
    let has_merges = model_dir.join("merges.txt").exists();
    let has_mel_filters = model_dir.join("mel_filters.bin").exists();

    let gguf_files = list_gguf_files(model_dir).unwrap_or_default();
    let preferred_gguf = asr_core::model_files::find_preferred_decoder_gguf(model_dir);

    let (safetensors_files, safetensors_error) =
        match asr_core::model_files::resolve_safetensors_files(model_dir) {
            Ok(v) => (Some(v), None),
            Err(e) => (None, Some(e.to_string())),
        };

    ModelInspection {
        model_dir: model_dir.to_path_buf(),
        has_config,
        has_vocab,
        has_merges,
        has_mel_filters,
        gguf_files,
        preferred_gguf,
        safetensors_files,
        safetensors_error,
    }
}

fn list_gguf_files(model_dir: &Path) -> std::io::Result<Vec<PathBuf>> {
    let mut out: Vec<PathBuf> = Vec::new();
    for entry in std::fs::read_dir(model_dir)? {
        let entry = entry?;
        let p = entry.path();
        if !p.is_file() {
            continue;
        }
        if p.extension().and_then(|s| s.to_str()) != Some("gguf") {
            continue;
        }
        out.push(p);
    }
    out.sort();
    Ok(out)
}

fn fmt_mib(bytes: u64) -> String {
    format!("{:.1} MiB", (bytes as f64) / (1024.0 * 1024.0))
}

fn run_models_list(root: PathBuf) -> Result<()> {
    use anyhow::Context;

    if !root.is_dir() {
        anyhow::bail!("Директория не найдена: {}", root.display());
    }

    let mut model_dirs: Vec<PathBuf> = Vec::new();
    for entry in std::fs::read_dir(&root).with_context(|| {
        format!(
            "Не удалось прочитать директорию моделей: {}",
            root.display()
        )
    })? {
        let entry = entry?;
        let p = entry.path();
        if p.is_dir() {
            model_dirs.push(p);
        }
    }
    model_dirs.sort();

    if model_dirs.is_empty() {
        println!(
            "В директории {} не найдено подпапок с моделями.",
            root.display()
        );
        return Ok(());
    }

    println!("Найдено моделей: {}", model_dirs.len());
    for dir in model_dirs {
        let insp = inspect_model_dir(&dir);
        let name = dir
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("<unknown>");

        let status = if insp.is_ready_for_transcribe() {
            "OK"
        } else {
            "PARTIAL"
        };

        let gguf = insp
            .preferred_gguf
            .as_ref()
            .and_then(|p| p.file_name())
            .and_then(|s| s.to_str())
            .unwrap_or("-");
        let gguf_count = insp.gguf_files.len();

        let st = match &insp.safetensors_files {
            Some(v) => format!("safetensors: {} file(s)", v.len()),
            None => "safetensors: -".to_string(),
        };

        println!("{name}: {status} ({st}, gguf: {gguf} ({gguf_count}))");
    }

    Ok(())
}

fn run_models_check(model_dir: PathBuf) -> Result<()> {
    use anyhow::Context;

    if !model_dir.is_dir() {
        anyhow::bail!("Директория модели не найдена: {}", model_dir.display());
    }

    let insp = inspect_model_dir(&model_dir);

    println!("Модель: {}", insp.model_dir.display());
    println!("Файлы:");
    println!(
        "- config.json: {}",
        if insp.has_config { "OK" } else { "MISSING" }
    );
    println!(
        "- vocab.json: {}",
        if insp.has_vocab { "OK" } else { "MISSING" }
    );
    println!(
        "- merges.txt: {}",
        if insp.has_merges { "OK" } else { "MISSING" }
    );
    println!(
        "- mel_filters.bin: {}",
        if insp.has_mel_filters {
            "OK"
        } else {
            "MISSING (будут использованы дефолтные mel-фильтры)"
        }
    );

    match &insp.safetensors_files {
        Some(files) => {
            let mut total = 0u64;
            for f in files {
                total += std::fs::metadata(f)
                    .with_context(|| format!("Не удалось получить metadata: {}", f.display()))?
                    .len();
            }
            println!(
                "- safetensors: OK ({} file(s), total {})",
                files.len(),
                fmt_mib(total)
            );
        }
        None => {
            println!("- safetensors: MISSING");
            if let Some(e) = &insp.safetensors_error {
                println!("  причина: {e}");
            }
        }
    }

    match &insp.preferred_gguf {
        Some(p) => {
            let sz = std::fs::metadata(p)
                .map(|m| fmt_mib(m.len()))
                .unwrap_or("-".to_string());
            println!(
                "- decoder gguf: OK (будет использован {}) ({sz})",
                p.file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or("<unknown>")
            );
        }
        None => {
            println!("- decoder gguf: MISSING (декодер будет загружен из safetensors)");
        }
    }

    if insp.gguf_files.is_empty() {
        println!("- gguf files: -");
    } else {
        let names: Vec<String> = insp
            .gguf_files
            .iter()
            .filter_map(|p| {
                p.file_name()
                    .and_then(|s| s.to_str())
                    .map(|s| s.to_string())
            })
            .collect();
        println!("- gguf files: {}", names.join(", "));
    }

    println!();
    println!(
        "Итог: transcribe={} | quantize={}",
        if insp.is_ready_for_transcribe() {
            "OK"
        } else {
            "NO"
        },
        if insp.is_ready_for_quantize() {
            "OK"
        } else {
            "NO"
        }
    );

    if !insp.is_ready_for_transcribe() {
        anyhow::bail!("Модель не готова для transcribe (см. вывод выше).");
    }

    Ok(())
}

fn estimate_max_tokens_for_segment(duration_sec: f32) -> usize {
    // Для сегментов VAD делаем меньший MIN, иначе короткие реплики легко уходят в генерацию мусора.
    const TOKENS_PER_SEC: f32 = 16.0;
    const OVERHEAD: usize = 32;
    const MIN: usize = 64;
    const MAX: usize = 2048;

    let est = (duration_sec * TOKENS_PER_SEC).ceil() as usize + OVERHEAD;
    est.clamp(MIN, MAX)
}

fn sanitize_filename(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        let ok = ch.is_ascii_alphanumeric() || ch == '_' || ch == '-' || ch == '.';
        out.push(if ok { ch } else { '_' });
    }
    if out.is_empty() {
        "speaker".to_string()
    } else {
        out
    }
}

fn extract_channel(buf: &asr_core::AudioBuffer, idx: usize) -> Result<asr_core::AudioBuffer> {
    if idx >= buf.channels {
        anyhow::bail!(
            "channel idx out of range: idx={}, channels={}",
            idx,
            buf.channels
        );
    }
    let mut out: Vec<f32> = Vec::with_capacity(buf.num_samples());
    for frame in buf.samples.chunks(buf.channels) {
        if let Some(&v) = frame.get(idx) {
            out.push(v);
        }
    }
    Ok(asr_core::AudioBuffer::new(out, buf.sample_rate, 1))
}

fn run_diarize(
    model: PathBuf,
    model_type: ModelTypeArg,
    audio_path: PathBuf,
    device_arg: &str,
    decoder_weights: DecoderWeightsArg,
    decoder_gguf: Option<PathBuf>,
    max_tokens: Option<usize>,
    language: Option<String>,
    speaker_language: Vec<String>,
    vad_mode: u8,
    vad_frame_ms: usize,
    vad_min_speech_ms: usize,
    vad_min_silence_ms: usize,
    vad_pad_ms: usize,
    vad_max_segment_s: usize,
    speaker_mode: SpeakerModeArg,
    num_speakers: Option<usize>,
    speaker_names: Vec<String>,
    left_speaker: &str,
    right_speaker: &str,
    right_num_speakers: Option<usize>,
    right_speaker_names: Vec<String>,
    out_dir: Option<PathBuf>,
) -> Result<()> {
    use anyhow::Context;
    use serde::Serialize;

    #[derive(Debug, Clone, Serialize)]
    struct JsonSegment {
        idx: usize,
        speaker: String,
        start_s: f32,
        end_s: f32,
        language: String,
        text: String,
        stop_reason: String,
    }

    #[derive(Debug, Clone)]
    struct WorkSegment {
        speaker: String,
        channel: usize,
        start: usize,
        end: usize,
    }

    /// Абстракция транскрибатора: Qwen3 (AsrPipeline) или любая другая модель (AsrEngine).
    enum Transcriber {
        Qwen3(asr_pipeline::AsrPipeline),
        Engine(asr_engine::AsrEngine),
    }

    struct SegmentResult {
        text: String,
        language: String,
        stop_reason: String,
    }

    impl Transcriber {
        fn transcribe_segment(
            &mut self,
            samples: &[f32],
            max_tokens: Option<usize>,
            language: Option<&str>,
            dur_s: f32,
        ) -> Result<SegmentResult> {
            match self {
                Transcriber::Qwen3(pipeline) => {
                    let mt = max_tokens.unwrap_or_else(|| estimate_max_tokens_for_segment(dur_s));
                    let r = pipeline.transcribe_to_result_with_max_tokens_and_language(
                        samples,
                        mt,
                        language,
                    )?;
                    let stop = match r.stop_reason {
                        asr_pipeline::StopReason::Eos => "eos",
                        asr_pipeline::StopReason::MaxTokens => "max_tokens",
                        asr_pipeline::StopReason::Repetition => "repetition",
                    };
                    Ok(SegmentResult {
                        text: r.text.trim().to_string(),
                        language: r.language.trim().to_string(),
                        stop_reason: stop.to_string(),
                    })
                }
                Transcriber::Engine(engine) => {
                    let mut options = asr_core::TranscribeOptions::default();
                    if let Some(lang) = language {
                        options = options.with_language(lang);
                    }
                    if let Some(mt) = max_tokens {
                        options = options.with_max_tokens(mt);
                    }
                    let r = engine.transcribe(samples, &options)?;
                    Ok(SegmentResult {
                        text: r.text.trim().to_string(),
                        language: r.language.unwrap_or_default(),
                        stop_reason: "eos".to_string(),
                    })
                }
            }
        }
    }

    const TARGET_SR: usize = 16000;

    let device = create_device(device_arg)?;

    // Создаём транскрибатор в зависимости от типа модели.
    let mut transcriber = match model_type {
        ModelTypeArg::Qwen3 => {
            let pipeline = asr_pipeline::AsrPipeline::from_model_dir_with_decoder_weights_and_gguf(
                &model,
                &device,
                decoder_weights.into(),
                decoder_gguf.clone(),
            )?;
            Transcriber::Qwen3(pipeline)
        }
        _ => {
            let quantized = matches!(decoder_weights, DecoderWeightsArg::Gguf);
            let engine = if quantized {
                asr_engine::AsrEngine::load_quantized(model_type.to_model_type(), &model, &device)?
            } else {
                asr_engine::AsrEngine::load(model_type.to_model_type(), &model, &device)?
            };
            Transcriber::Engine(engine)
        }
    };

    let start_all = Instant::now();

    let audio_buf = audio::load_wav(&audio_path)?;
    let audio_sample_rate = audio_buf.sample_rate;
    let audio_channels = audio_buf.channels;
    let audio_duration = audio_buf.duration();
    println!("🎧 Diarize + VAD transcription");
    println!("================================");
    println!("Model: {}", model.display());
    println!("Model type: {:?}", model_type);
    println!("Audio: {}", audio_path.display());
    println!(
        "Audio: {} Hz, channels={}, duration={:.2}s",
        audio_sample_rate, audio_channels, audio_duration
    );
    println!("Device: {}", device_arg);
    println!();

    if audio_channels == 0 {
        anyhow::bail!("Некорректный WAV: channels=0");
    }
    if audio_channels > 2 {
        anyhow::bail!(
            "Пока поддерживается только mono/stereo WAV (channels={})",
            audio_channels
        );
    }

    let resampler = Resampler::new(TARGET_SR);
    let mut chan_samples: Vec<Vec<f32>> = Vec::new();
    let mut chan_speakers: Vec<String> = Vec::new();

    let effective_speaker_mode = match speaker_mode {
        SpeakerModeArg::Auto => {
            if audio_channels >= 2 {
                SpeakerModeArg::Channel
            } else {
                SpeakerModeArg::Cluster
            }
        }
        SpeakerModeArg::Channel => SpeakerModeArg::Channel,
        SpeakerModeArg::Cluster => SpeakerModeArg::Cluster,
    };

    if audio_channels == 1 {
        let mono = audio_buf;
        let mono = if mono.sample_rate != TARGET_SR {
            resampler
                .resample(&mono)
                .context("Resampling failed for mono audio")?
        } else {
            mono
        };
        chan_samples.push(mono.samples);
        chan_speakers.push(left_speaker.to_string());
    } else {
        let left = extract_channel(&audio_buf, 0)?;
        let right = extract_channel(&audio_buf, 1)?;
        let left = if left.sample_rate != TARGET_SR {
            resampler
                .resample(&left)
                .context("Resampling failed for left channel")?
        } else {
            left
        };
        let right = if right.sample_rate != TARGET_SR {
            resampler
                .resample(&right)
                .context("Resampling failed for right channel")?
        } else {
            right
        };

        match effective_speaker_mode {
            SpeakerModeArg::Channel => {
                chan_samples.push(left.samples);
                chan_samples.push(right.samples);
                chan_speakers.push(left_speaker.to_string());
                chan_speakers.push(right_speaker.to_string());
            }
            SpeakerModeArg::Cluster | SpeakerModeArg::Auto => {
                // Для кластеризации лучше работать с mono-миксом, чтобы не удваивать сегменты.
                let n = left.samples.len().min(right.samples.len());
                let mut mix: Vec<f32> = Vec::with_capacity(n);
                for i in 0..n {
                    mix.push(0.5 * (left.samples[i] + right.samples[i]));
                }
                chan_samples.push(mix);
                chan_speakers.push("mix".to_string());
            }
        }
    }

    let vad_cfg = vad::VadSegmentationConfig {
        mode: vad_mode,
        frame_ms: vad_frame_ms,
        min_speech_ms: vad_min_speech_ms,
        min_silence_ms: vad_min_silence_ms,
        pad_ms: vad_pad_ms,
        max_segment_ms: vad_max_segment_s.saturating_mul(1000),
    };

    let speaker_names: Vec<String> = speaker_names
        .into_iter()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();

    let right_speaker_names: Vec<String> = right_speaker_names
        .into_iter()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();

    let mut right_channel_clustering: Option<String> = None;

    let mut segments: Vec<WorkSegment> = Vec::new();
    match effective_speaker_mode {
        SpeakerModeArg::Channel => {
            for (ch_idx, samples) in chan_samples.iter().enumerate() {
                let speech = vad::split_mono_by_vad(samples.as_slice(), TARGET_SR, &vad_cfg)
                    .with_context(|| format!("VAD failed for channel {ch_idx}"))?;
                let spk = chan_speakers[ch_idx].clone();
                println!(
                    "VAD: channel={} speaker={} -> {} segments",
                    ch_idx,
                    spk,
                    speech.len()
                );
                for s in speech {
                    segments.push(WorkSegment {
                        speaker: spk.clone(),
                        channel: ch_idx,
                        start: s.start_sample,
                        end: s.end_sample,
                    });
                }
            }
        }
        SpeakerModeArg::Cluster | SpeakerModeArg::Auto => {
            // В cluster-режиме у нас всегда 1 "канал" (mono).
            let samples = chan_samples
                .get(0)
                .context("internal error: chan_samples is empty")?;
            let speech = vad::split_mono_by_vad(samples.as_slice(), TARGET_SR, &vad_cfg)
                .context("VAD failed for mono audio")?;
            println!("VAD: speaker-mode=cluster -> {} segments", speech.len());

            for s in speech {
                segments.push(WorkSegment {
                    speaker: "spk?".to_string(),
                    channel: 0,
                    start: s.start_sample,
                    end: s.end_sample,
                });
            }

            let k = num_speakers.unwrap_or(2);
            if k == 0 {
                anyhow::bail!("num-speakers=0 недопустим (ожидается >= 1)");
            }

            if segments.len() >= 2 && k >= 2 {
                let bounds: Vec<(usize, usize)> =
                    segments.iter().map(|s| (s.start, s.end)).collect();
                let assign = speaker::cluster_segments_mel_kmeans(
                    &model,
                    TARGET_SR,
                    samples.as_slice(),
                    bounds.as_slice(),
                    k.min(bounds.len()),
                )?;
                println!(
                    "Speaker clustering: k={} (names={})",
                    k.min(bounds.len()),
                    if speaker_names.is_empty() {
                        "-"
                    } else {
                        "provided"
                    }
                );

                for (seg, spk_id) in segments.iter_mut().zip(assign.into_iter()) {
                    let label = speaker_names
                        .get(spk_id)
                        .cloned()
                        .unwrap_or_else(|| format!("spk{spk_id}"));
                    seg.speaker = label;
                }
            } else {
                let label = speaker_names
                    .get(0)
                    .cloned()
                    .unwrap_or_else(|| "spk0".to_string());
                for seg in &mut segments {
                    seg.speaker = label.clone();
                }
            }
        }
    }

    // Дополнительная диаризация внутри правого канала (system), если там несколько собеседников.
    if matches!(effective_speaker_mode, SpeakerModeArg::Channel)
        && audio_channels == 2
        && right_num_speakers.unwrap_or(0) >= 2
    {
        let k_req = right_num_speakers.unwrap_or(0);
        let right_ch = 1usize;
        let right_samples = chan_samples
            .get(right_ch)
            .context("internal error: right channel samples missing")?;

        let mut right_indices: Vec<usize> = Vec::new();
        let mut bounds: Vec<(usize, usize)> = Vec::new();
        for (i, seg) in segments.iter().enumerate() {
            if seg.channel == right_ch {
                right_indices.push(i);
                bounds.push((seg.start, seg.end));
            }
        }

        if bounds.len() >= 2 {
            let k = k_req.min(bounds.len());
            let assign = speaker::cluster_segments_mel_kmeans(
                &model,
                TARGET_SR,
                right_samples.as_slice(),
                bounds.as_slice(),
                k,
            )?;

            let mut labels: Vec<String> = Vec::with_capacity(k);
            for spk_id in 0..k {
                let label = match right_speaker_names.get(spk_id) {
                    Some(name) => format!("{right_speaker}-{name}"),
                    None => format!("{right_speaker}-spk{spk_id}"),
                };
                labels.push(label);
            }

            for (idx, spk_id) in right_indices.into_iter().zip(assign.into_iter()) {
                if let Some(lbl) = labels.get(spk_id) {
                    segments[idx].speaker = lbl.clone();
                }
            }

            println!(
                "Right channel speaker clustering: k={} -> {}",
                k,
                labels.join(", ")
            );
            right_channel_clustering = Some(format!(
                "enabled (k={k}, base={right_speaker}, names={})",
                if right_speaker_names.is_empty() {
                    "-"
                } else {
                    "provided"
                }
            ));
        } else {
            println!(
                "Right channel speaker clustering requested (k={}) but skipped: only {} segment(s)",
                k_req,
                bounds.len()
            );
            right_channel_clustering = Some(format!(
                "requested (k={k_req}) but skipped (segments={})",
                bounds.len()
            ));
        }
    } else if right_num_speakers.is_some() {
        right_channel_clustering =
            Some("requested but not applicable (need stereo + speaker-mode=channel)".to_string());
    }

    if segments.is_empty() {
        anyhow::bail!("VAD не нашел сегментов речи (проверьте параметры vad-*)");
    }

    segments.sort_by_key(|s| s.start);

    let out_dir = match out_dir {
        Some(p) => p,
        None => vad::default_out_dir("docs/transcriptions/diarize")?,
    };
    std::fs::create_dir_all(&out_dir)
        .with_context(|| format!("Не удалось создать out-dir: {}", out_dir.display()))?;

    let report_path = out_dir.join("report.md");
    let combined_path = out_dir.join("combined.txt");
    let json_path = out_dir.join("segments.json");

    println!();
    println!("✍️  Transcribing segments: {}", segments.len());
    println!("Out dir: {}", out_dir.display());
    println!();

    let mut combined_lines: Vec<String> = Vec::with_capacity(segments.len());
    let mut json_segments: Vec<JsonSegment> = Vec::with_capacity(segments.len());
    let mut per_speaker: std::collections::BTreeMap<String, String> =
        std::collections::BTreeMap::new();
    let mut speaker_language_overrides: std::collections::BTreeMap<String, String> =
        std::collections::BTreeMap::new();

    for spec in speaker_language {
        let (speaker, lang) = spec.split_once('=').with_context(|| {
            format!("Некорректный --speaker-language (ожидается SPEAKER=LANG): {spec}")
        })?;
        let speaker = speaker.trim();
        let lang = lang.trim();
        if speaker.is_empty() {
            anyhow::bail!("Некорректный --speaker-language: пустой SPEAKER: {spec}");
        }
        if lang.is_empty() {
            anyhow::bail!("Некорректный --speaker-language: пустой LANG: {spec}");
        }
        speaker_language_overrides.insert(speaker.to_string(), lang.to_string());
    }

    for (idx, seg) in segments.iter().enumerate() {
        let s = seg.start.min(chan_samples[seg.channel].len());
        let e = seg.end.min(chan_samples[seg.channel].len());
        if e <= s {
            continue;
        }

        let seg_samples = &chan_samples[seg.channel][s..e];
        let start_s = s as f32 / TARGET_SR as f32;
        let end_s = e as f32 / TARGET_SR as f32;
        let dur_s = (e - s) as f32 / TARGET_SR as f32;

        let forced_lang = speaker_language_overrides
            .get(&seg.speaker)
            .map(|s| s.as_str())
            .or_else(|| {
                // Удобный fallback: если спикер имеет вид `system-spk0`, то
                // разрешаем задавать override как `--speaker-language system=English`.
                seg.speaker
                    .split_once('-')
                    .and_then(|(prefix, _)| speaker_language_overrides.get(prefix))
                    .map(|s| s.as_str())
            })
            .or(language.as_deref());
        let r = transcriber.transcribe_segment(
            seg_samples,
            max_tokens,
            forced_lang,
            dur_s,
        )?;

        let ts0 = vad::format_hhmmss_millis(start_s);
        let ts1 = vad::format_hhmmss_millis(end_s);

        combined_lines.push(format!("[{ts0} - {ts1}] {}: {}", seg.speaker, r.text));
        per_speaker
            .entry(seg.speaker.clone())
            .or_default()
            .push_str(&format!("{}\n", r.text));

        json_segments.push(JsonSegment {
            idx,
            speaker: seg.speaker.clone(),
            start_s,
            end_s,
            language: r.language,
            text: r.text,
            stop_reason: r.stop_reason.clone(),
        });

        if r.stop_reason == "max_tokens" {
            eprintln!(
                "⚠️  segment idx={} speaker={} reached max_tokens ({}-{})",
                idx, seg.speaker, ts0, ts1
            );
        } else if r.stop_reason == "repetition" {
            eprintln!(
                "⚠️  segment idx={} speaker={} stopped by repetition heuristic ({}-{})",
                idx, seg.speaker, ts0, ts1
            );
        }
    }

    std::fs::write(&combined_path, combined_lines.join("\n"))?;
    std::fs::write(&json_path, serde_json::to_string_pretty(&json_segments)?)?;

    for (speaker, text) in &per_speaker {
        let fname = sanitize_filename(speaker);
        let p = out_dir.join(format!("{fname}.txt"));
        std::fs::write(&p, text.as_bytes())?;
    }

    // Простой markdown-отчет.
    let mut md = String::new();
    md.push_str("# Диаризация (VAD) + транскрибация по сегментам\n\n");
    md.push_str(&format!("- Model: `{}`\n", model.display()));
    md.push_str(&format!("- Audio: `{}`\n", audio_path.display()));
    md.push_str(&format!("- Device: `{}`\n", device_arg));
    md.push_str(&format!(
        "- Decoder: weights=`{}` gguf=`{}`\n",
        match decoder_weights {
            DecoderWeightsArg::Auto => "auto",
            DecoderWeightsArg::Safetensors => "safetensors",
            DecoderWeightsArg::Gguf => "gguf",
        },
        decoder_gguf
            .as_ref()
            .and_then(|p| p.to_str())
            .unwrap_or("-")
    ));
    md.push_str(&format!(
        "- Language: `{}`\n",
        language.as_deref().unwrap_or("auto")
    ));
    if speaker_language_overrides.is_empty() {
        md.push_str("- Speaker language overrides: `-`\n");
    } else {
        let items: Vec<String> = speaker_language_overrides
            .iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect();
        md.push_str(&format!(
            "- Speaker language overrides: `{}`\n",
            items.join(", ")
        ));
    }
    md.push_str(&format!(
        "- Speaker mode: `{}`\n",
        match effective_speaker_mode {
            SpeakerModeArg::Auto => "auto",
            SpeakerModeArg::Channel => "channel",
            SpeakerModeArg::Cluster => "cluster",
        }
    ));
    md.push_str(&format!(
        "- Right channel clustering: `{}`\n",
        right_channel_clustering.as_deref().unwrap_or("-")
    ));
    if matches!(
        effective_speaker_mode,
        SpeakerModeArg::Cluster | SpeakerModeArg::Auto
    ) {
        md.push_str(&format!("- Num speakers: {}\n", num_speakers.unwrap_or(2)));
    }
    md.push_str(&format!(
        "- Audio info: {} Hz, channels={}, duration={:.2}s\n",
        audio_sample_rate, audio_channels, audio_duration
    ));
    md.push_str(&format!(
        "- VAD: mode={}, frame_ms={}, min_speech_ms={}, min_silence_ms={}, pad_ms={}, max_segment_s={}\n",
        vad_mode, vad_frame_ms, vad_min_speech_ms, vad_min_silence_ms, vad_pad_ms, vad_max_segment_s
    ));
    md.push_str(&format!("- Segments: {}\n", json_segments.len()));
    md.push_str(&format!(
        "- Outputs: `{}`, `{}`, `{}`\n\n",
        report_path.display(),
        combined_path.display(),
        json_path.display()
    ));

    md.push_str("## Таблица сегментов\n\n");
    md.push_str("| # | Start | End | Speaker | Lang | Stop | Text |\n");
    md.push_str("|---:|---|---|---|---|---|---|\n");
    for s in &json_segments {
        let ts0 = vad::format_hhmmss_millis(s.start_s);
        let ts1 = vad::format_hhmmss_millis(s.end_s);
        let text = s.text.replace('|', "\\|");
        let lang = s.language.replace('|', "\\|");
        md.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | {} |\n",
            s.idx, ts0, ts1, s.speaker, lang, s.stop_reason, text
        ));
    }

    md.push_str("\n## Склейка (по времени)\n\n");
    md.push_str("См. также `combined.txt`.\n\n```text\n");
    md.push_str(&combined_lines.join("\n"));
    md.push_str("\n```\n");

    md.push_str(&format!(
        "\nВремя выполнения: {:.2}s\n",
        start_all.elapsed().as_secs_f32()
    ));

    std::fs::write(&report_path, md.as_bytes())?;

    println!("✅ Done");
    println!("Report: {}", report_path.display());
    println!("Combined: {}", combined_path.display());
    println!("JSON: {}", json_path.display());

    Ok(())
}

fn create_device(device: &str) -> Result<candle_core::Device> {
    match device {
        "metal" => {
            // Настроить параметры Metal command buffer pool для стабильности.
            // Workaround для AGXMetalG16X::fillBuffer SIGSEGV на M4 / macOS 26.x.
            asr_core::metal_utils::configure_metal_env();

            // candle может panic в процессе инициализации Metal (например, если устройство недоступно).
            // Панику ловим, а hook временно глушим, чтобы не засорять stderr.
            let prev_hook = std::panic::take_hook();
            std::panic::set_hook(Box::new(|_| {}));
            let res = std::panic::catch_unwind(|| candle_core::Device::new_metal(0));
            std::panic::set_hook(prev_hook);

            match res {
                Ok(Ok(dev)) => Ok(dev),
                Ok(Err(e)) => Err(e.into()),
                Err(_) => Err(anyhow::anyhow!(
                    "Инициализация Metal недоступна в этом окружении. Попробуйте --device cpu."
                )),
            }
        }
        "cuda" => Ok(candle_core::Device::new_cuda(0)?),
        _ => Ok(candle_core::Device::Cpu),
    }
}

fn run_quantize(
    input: PathBuf,
    output: PathBuf,
    qtype: &str,
    scope: &str,
    quantize_embeddings: bool,
) -> Result<()> {
    use anyhow::Context;
    use candle_core::quantized::{GgmlDType, QTensor, gguf_file};
    use candle_core::{DType, Device};
    use tracing::info;

    let start = Instant::now();
    let (input_desc, safetensors_files) = if input.is_dir() {
        let files = asr_core::model_files::resolve_safetensors_files(&input)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;
        (input.display().to_string(), files)
    } else {
        if !input.exists() {
            anyhow::bail!("Input file not found: {:?}", input);
        }

        if input.file_name().and_then(|s| s.to_str()) == Some("model.safetensors.index.json") {
            let dir = input
                .parent()
                .context("model.safetensors.index.json has no parent dir")?;
            let files = asr_core::model_files::resolve_safetensors_files(dir)
                .map_err(|e| anyhow::anyhow!(e.to_string()))?;
            (dir.display().to_string(), files)
        } else {
            (input.display().to_string(), vec![input])
        }
    };

    let ggml = match qtype {
        "q8_0" => GgmlDType::Q8_0,
        "q6k" | "q6_k" => GgmlDType::Q6K,
        "q4_0" => GgmlDType::Q4_0,
        other => anyhow::bail!("Unsupported qtype: {other} (expected: q8_0, q6k, q4_0)"),
    };

    info!(
        input = %input_desc,
        output = ?output,
        qtype = %qtype,
        scope = %scope,
        quantize_embeddings,
        shards = safetensors_files.len(),
        "Starting quantization"
    );

    let device = Device::Cpu;
    let mut out_tensors: Vec<(String, QTensor)> = Vec::new();
    let mut num_quantized = 0usize;
    let mut num_kept = 0usize;

    for f in safetensors_files {
        let tensors = candle_core::safetensors::load(&f, &device)
            .with_context(|| format!("Failed to load safetensors shard: {}", f.display()))?;

        for (name, tensor) in tensors {
            if !name.starts_with(scope) {
                continue;
            }

            let is_embedding =
                name.ends_with("embed_tokens.weight") || name.contains("embed_tokens.weight");
            let is_norm =
                name.contains("norm") || name.contains("layernorm") || name.contains("rms_norm");

            let should_quantize = tensor.rank() == 2
                && name.ends_with(".weight")
                && !is_norm
                && (!is_embedding || quantize_embeddings);

            if should_quantize {
                let qt = QTensor::quantize(&tensor, ggml)?;
                out_tensors.push((name, qt));
                num_quantized += 1;
            } else {
                // Для экономии памяти по умолчанию храним "не-квантованные" тензоры как F16.
                // (Нормы/позиционки/мелкие веса в F16 обычно безопасны.)
                let t16 = if tensor.dtype() == DType::F16 {
                    tensor
                } else {
                    tensor.to_dtype(DType::F16)?
                };
                let qt = QTensor::quantize(&t16, GgmlDType::F16)?;
                out_tensors.push((name, qt));
                num_kept += 1;
            }
        }
    }

    info!(num_quantized, num_kept, "Prepared tensors");

    let mut f = std::fs::File::create(&output)?;
    let refs: Vec<(&str, &QTensor)> = out_tensors.iter().map(|(n, t)| (n.as_str(), t)).collect();
    gguf_file::write(&mut f, &[], &refs)?;

    info!(elapsed = ?start.elapsed(), "Quantization complete");
    Ok(())
}
