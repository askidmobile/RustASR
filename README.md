# RustASR: Qwen3-based Automatic Speech Recognition

## Описание проекта
**RustASR** — это экспериментальный проект по реализации системы автоматического распознавания речи (ASR) на языке Rust с использованием фреймворка `candle`.
Проект базируется на архитектуре модели **Qwen3-ASR** (Qwen2-Audio), которая использует специализированный аудио-энкодер AuT (Attention-based Encoder) и LLM Qwen3.

## Цель
Создать эффективную, быструю и безопасную реализацию ASR, способную работать локально на CPU/GPU (Metal), полностью на Rust.

## Ссылки и Референсы
*   **Референсный проект**: [RustTTS](../RustTTS) (наш проект синтеза речи). Мы заимствуем из него подходы к структуре, работе с `candle` и моделью Qwen3.
*   **Оригинальная модель**: [QwenLM/Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR).

## Документация
Полное описание требований и архитектуры находится в **[Product Requirements Document (PRD)](docs/PRD.md)**.

Детальные планы по фазам:
*   [Фаза 1: Инфраструктура](docs/PHASE_1_INFRASTRUCTURE.md)
*   [Фаза 2: Обработка Аудио](docs/PHASE_2_FEATURE_EXTRACTOR.md)
*   [Фаза 3: AuT Энкодер](docs/PHASE_3_AUT_ENCODER.md)
*   [Фаза 4: Интеграция](docs/PHASE_4_INTEGRATION.md)
