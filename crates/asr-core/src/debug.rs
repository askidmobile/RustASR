//! Вспомогательные функции для отладочного вывода.

use std::sync::OnceLock;

/// Возвращает `true`, если включен подробный отладочный вывод.
///
/// Управляется переменной окружения `RUSTASR_DEBUG` (любое непустое значение).
pub fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("RUSTASR_DEBUG").is_some())
}
