#!/usr/bin/env python3
"""
Создание тестового WAV файла с синусоидой для верификации.
"""

import numpy as np
import soundfile as sf
from pathlib import Path

def create_test_wav(
    path: str,
    duration: float = 1.0,
    sample_rate: int = 16000,
    frequency: float = 440.0,
):
    """Создание тестового WAV файла с синусоидой."""
    t = np.linspace(0, duration, int(duration * sample_rate), dtype=np.float32)
    audio = 0.5 * np.sin(2 * np.pi * frequency * t)
    sf.write(path, audio, sample_rate)
    print(f"✅ Создан: {path}")
    print(f"   Длительность: {duration}s, Частота: {frequency}Hz")
    return audio

if __name__ == "__main__":
    output_dir = Path(__file__).parent.parent / "tests" / "fixtures"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Создаём тестовые файлы
    create_test_wav(str(output_dir / "test_sine_440hz.wav"), duration=1.0, frequency=440.0)
    create_test_wav(str(output_dir / "test_sine_1khz.wav"), duration=0.5, frequency=1000.0)
    
    print(f"\n📁 Файлы созданы в: {output_dir}")
