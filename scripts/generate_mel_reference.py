#!/usr/bin/env python3
"""
Генерация эталонных Mel спектрограмм для верификации RustASR.

Использование:
    python generate_mel_reference.py <audio.wav> --output reference.npy
    
Зависимости:
    pip install numpy librosa soundfile
"""

import argparse
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path


def load_audio(path: str, target_sr: int = 16000) -> np.ndarray:
    """Загрузка и ресемплинг аудио."""
    audio, sr = sf.read(path)
    
    # Конвертация в моно
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
    
    # Ресемплинг до целевой частоты
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    
    return audio.astype(np.float32)


def extract_mel_spectrogram(
    audio: np.ndarray,
    sample_rate: int = 16000,
    n_fft: int = 400,
    hop_length: int = 160,
    n_mels: int = 128,
    f_min: float = 0.0,
    f_max: float = 8000.0,
) -> np.ndarray:
    """
    Извлечение Log-Mel спектрограммы.
    
    Параметры соответствуют Qwen3-ASR:
    - sample_rate: 16000 Hz
    - n_fft: 400 (25ms window)
    - hop_length: 160 (10ms hop)
    - n_mels: 128
    """
    # Вычисление Mel спектрограммы
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=f_min,
        fmax=f_max,
        power=1.0,  # Magnitude spectrogram
    )
    
    # Логарифмирование
    log_mel = np.log(np.maximum(mel_spec, 1e-10))
    
    # Транспонирование: [n_mels, time] -> [time, n_mels]
    log_mel = log_mel.T
    
    return log_mel.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(
        description="Генерация эталонных Mel спектрограмм для RustASR"
    )
    parser.add_argument("audio", type=str, help="Путь к WAV файлу")
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Путь для сохранения .npy файла (по умолчанию: <audio>_mel.npy)"
    )
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--n-fft", type=int, default=400)
    parser.add_argument("--hop-length", type=int, default=160)
    parser.add_argument("--n-mels", type=int, default=128)
    
    args = parser.parse_args()
    
    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"Ошибка: файл {audio_path} не найден")
        return 1
    
    output_path = args.output or str(audio_path.with_suffix("")) + "_mel.npy"
    
    print(f"📂 Загрузка: {audio_path}")
    audio = load_audio(str(audio_path), args.sample_rate)
    print(f"   Длительность: {len(audio) / args.sample_rate:.2f}s")
    
    print(f"📊 Извлечение Mel спектрограммы...")
    mel = extract_mel_spectrogram(
        audio,
        sample_rate=args.sample_rate,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        n_mels=args.n_mels,
    )
    print(f"   Форма: {mel.shape} [time, n_mels]")
    
    print(f"💾 Сохранение: {output_path}")
    np.save(output_path, mel)
    
    # Также сохраним сырое аудио для тестирования Rust
    audio_npy_path = str(audio_path.with_suffix("")) + "_audio.npy"
    np.save(audio_npy_path, audio)
    print(f"💾 Аудио сохранено: {audio_npy_path}")
    
    print(f"\n✅ Готово! Используйте эти файлы для верификации Rust:")
    print(f"   - Mel: {output_path}")
    print(f"   - Audio: {audio_npy_path}")
    
    # Выводим статистику для отладки
    print(f"\n📈 Статистика Mel спектрограммы:")
    print(f"   Min: {mel.min():.4f}")
    print(f"   Max: {mel.max():.4f}")
    print(f"   Mean: {mel.mean():.4f}")
    print(f"   Std: {mel.std():.4f}")
    
    return 0


if __name__ == "__main__":
    exit(main())
