//! Утилиты для безопасной работы с Apple Metal GPU.
//!
//! Apple Metal driver на некоторых комбинациях macOS / чипов (M4, macOS 26.x)
//! содержит баг, приводящий к SIGSEGV в `AGXMetalG16X::fillBuffer` при
//! определённых паттернах переиспользования буферов в command buffer pool.
//!
//! Этот модуль предоставляет workaround'ы:
//! - [`configure_metal_env`] — настройка переменных окружения candle для более
//!   консервативного управления Metal command buffers.
//! - [`metal_probe`] — пробное вычисление на GPU для проверки работоспособности.
//! - [`metal_sync`] — барьер синхронизации для сброса pending операций.

use candle_core::{DType, Device, Tensor};
use tracing::{debug, info, warn};

use crate::{AsrError, AsrResult};

/// Максимальное количество compute encoder'ов на один command buffer.
///
/// Значение по умолчанию в candle (высокое) приводит к накоплению большого
/// количества операций в одном command buffer, что увеличивает вероятность
/// бага с `fillBuffer` на уязвимых драйверах.
///
/// Уменьшение до 16 даёт более частый commit/flush и снижает риск краша
/// при минимальном влиянии на производительность (порядок микросекунд).
const SAFE_COMPUTE_PER_BUFFER: &str = "16";

/// Настроить переменные окружения candle Metal для стабильной работы.
///
/// **Вызывать ДО создания `Device::new_metal(...)`**, т.к. candle читает
/// переменные окружения при инициализации command pool.
///
/// Не перезаписывает переменные, если пользователь уже задал их явно.
pub fn configure_metal_env() {
    // Уменьшить количество compute encoder'ов на command buffer.
    if std::env::var("CANDLE_METAL_COMPUTE_PER_BUFFER").is_err() {
        // SAFETY: вызывается до создания Metal device, потоков-потребителей нет.
        unsafe {
            std::env::set_var("CANDLE_METAL_COMPUTE_PER_BUFFER", SAFE_COMPUTE_PER_BUFFER);
        }
        debug!(
            "Metal: CANDLE_METAL_COMPUTE_PER_BUFFER={}",
            SAFE_COMPUTE_PER_BUFFER
        );
    }
}

/// Запустить пробное вычисление на Metal device для проверки работоспособности.
///
/// Тестирует полный цикл: аллокация буфера, compute kernel (matmul),
/// blit encoder (fill_buffer через Tensor::zeros), readback на CPU.
///
/// Если проба проходит без ошибок — Metal device можно безопасно использовать.
/// При ошибке возвращает `AsrError` с описанием проблемы.
///
/// # Пример
/// ```no_run
/// # use candle_core::Device;
/// # use asr_core::metal_utils;
/// let device = Device::new_metal(0).unwrap();
/// if let Err(e) = metal_utils::metal_probe(&device) {
///     eprintln!("Metal probe failed: {e}, falling back to CPU");
///     // ... использовать Device::Cpu
/// }
/// ```
pub fn metal_probe(device: &Device) -> AsrResult<()> {
    if !device.is_metal() {
        return Ok(());
    }

    info!("Metal: запуск пробного вычисления для проверки GPU...");

    // 1. Тест compute: создать тензоры из данных, matmul, readback.
    //    Tensor::from_vec использует newBufferWithBytes (безопасная копия).
    let a = Tensor::from_vec(vec![1.0f32; 64], (8, 8), device).map_err(|e| {
        AsrError::Device(format!("Metal probe: не удалось создать тензор A: {e}"))
    })?;
    let b = Tensor::from_vec(vec![1.0f32; 64], (8, 8), device).map_err(|e| {
        AsrError::Device(format!("Metal probe: не удалось создать тензор B: {e}"))
    })?;
    let c = a.matmul(&b).map_err(|e| {
        AsrError::Device(format!("Metal probe: matmul failed: {e}"))
    })?;
    // Синхронизация + readback: тестирует blit encoder (copy_from_buffer).
    device.synchronize().map_err(|e| {
        AsrError::Device(format!("Metal probe: synchronize after matmul failed: {e}"))
    })?;
    let _data: Vec<Vec<f32>> = c.to_vec2().map_err(|e| {
        AsrError::Device(format!("Metal probe: readback (matmul) failed: {e}"))
    })?;

    // 2. Тест allocate_zeros: Tensor::zeros использует blit fill_buffer —
    //    именно эта операция вызывает краш на уязвимых драйверах.
    let z = Tensor::zeros((8, 8), DType::F32, device).map_err(|e| {
        AsrError::Device(format!("Metal probe: Tensor::zeros failed: {e}"))
    })?;
    device.synchronize().map_err(|e| {
        AsrError::Device(format!("Metal probe: synchronize after zeros failed: {e}"))
    })?;
    let _zdata: Vec<Vec<f32>> = z.to_vec2().map_err(|e| {
        AsrError::Device(format!("Metal probe: readback (zeros) failed: {e}"))
    })?;

    // 3. Тест Conv1d-подобных операций: линейное преобразование.
    //    GigaAM активно использует Conv1d, проверяем аналогичные паттерны.
    let x = Tensor::from_vec(vec![0.5f32; 768], (1, 768), device).map_err(|e| {
        AsrError::Device(format!("Metal probe: не удалось создать тензор для linear test: {e}"))
    })?;
    let w = Tensor::from_vec(vec![0.1f32; 768 * 768], (768, 768), device).map_err(|e| {
        AsrError::Device(format!("Metal probe: не удалось создать тензор весов: {e}"))
    })?;
    let y = x.matmul(&w.t()?).map_err(|e| {
        AsrError::Device(format!("Metal probe: large matmul failed: {e}"))
    })?;
    device.synchronize().map_err(|e| {
        AsrError::Device(format!("Metal probe: synchronize after large matmul failed: {e}"))
    })?;
    let _ydata: Vec<Vec<f32>> = y.to_vec2().map_err(|e| {
        AsrError::Device(format!("Metal probe: readback (large matmul) failed: {e}"))
    })?;

    info!("Metal: пробное вычисление успешно — GPU работоспособен");
    Ok(())
}

/// Синхронизация Metal command buffers.
///
/// Сбрасывает все pending GPU-операции и ждёт их завершения.
/// На не-Metal устройствах — no-op.
///
/// Используется как барьер между фазами инференса:
/// - после загрузки данных на GPU (mel → Metal);
/// - между группами слоёв encoder'а;
/// - перед финальным readback на CPU.
///
/// Издержки: ~50-200 мкс на вызов (зависит от кол-ва pending ops).
#[inline]
pub fn metal_sync(device: &Device) -> AsrResult<()> {
    if device.is_metal() {
        device.synchronize().map_err(|e| {
            AsrError::Device(format!("Metal sync failed: {e}"))
        })?;
    }
    Ok(())
}

/// Создать Metal device с предварительной проверкой.
///
/// 1. Настраивает переменные окружения Metal.
/// 2. Создаёт Metal device.
/// 3. Запускает [`metal_probe`] для проверки работоспособности.
///
/// При неудаче возвращает `Err` с рекомендацией использовать CPU.
pub fn create_safe_metal_device() -> AsrResult<Device> {
    configure_metal_env();

    // Инициализация Metal может panic (candle).
    let prev_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let result = std::panic::catch_unwind(|| Device::new_metal(0));
    std::panic::set_hook(prev_hook);

    let device = match result {
        Ok(Ok(dev)) => dev,
        Ok(Err(e)) => {
            return Err(AsrError::Device(format!(
                "Metal device недоступен: {e}. Используйте --device cpu."
            )));
        }
        Err(_) => {
            return Err(AsrError::Device(
                "Panic при инициализации Metal. Используйте --device cpu.".into(),
            ));
        }
    };

    // Пробное вычисление — если GPU драйвер нестабилен, упадёт здесь.
    match metal_probe(&device) {
        Ok(()) => Ok(device),
        Err(e) => {
            warn!("Metal probe провалился: {e}");
            Err(AsrError::Device(format!(
                "Metal GPU не прошёл проверку: {e}. Используйте --device cpu."
            )))
        }
    }
}
