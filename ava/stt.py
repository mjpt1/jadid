"""رونویسی گفتار به متن با faster-whisper (کاملاً لوکال)."""

from __future__ import annotations

import numpy as np
from faster_whisper import WhisperModel

_CACHE: dict[str, WhisperModel] = {}


def _get_model(name: str, device: str, compute_type: str) -> WhisperModel:
    key = f"{name}|{device}|{compute_type}"
    if key not in _CACHE:
        _CACHE[key] = WhisperModel(name, device=device, compute_type=compute_type)
    return _CACHE[key]


class STT:
    def __init__(self, model: str, device: str = "auto", compute_type: str = "int8", language: str = "fa"):
        self.language = language
        self.model = _get_model(model, device, compute_type)

    @staticmethod
    def _to_float(audio_int16: np.ndarray) -> np.ndarray:
        audio = audio_int16.astype(np.float32)
        if not audio.size:
            return audio
        # اگر ورودی int16 است (نه float نرمال‌شده) کنترل خودکار بهره اعمال کن
        peak = float(np.max(np.abs(audio)))
        if peak > 1.0:
            cur_rms = float(np.sqrt(np.mean(audio**2))) or 1.0
            target_rms = 0.08 * 32768.0          # هدف: حدود ۸٪ مقیاس کامل
            gain = min(target_rms / cur_rms, 32768.0 / max(peak, 1.0))  # بدون کلیپ
            gain = min(gain, 30.0)               # سقف تقویت تا نویز منفجر نشود
            audio = audio * gain
        return audio / 32768.0

    def transcribe(
        self,
        audio_int16: np.ndarray,
        vad_filter: bool = False,
        beam_size: int = 1,
        permissive: bool = False,
    ) -> str:
        if audio_int16 is None or audio_int16.size == 0:
            return ""
        kwargs = dict(
            language=self.language,
            vad_filter=vad_filter,
            beam_size=beam_size,
            condition_on_previous_text=False,
            without_timestamps=True,
        )
        if permissive:
            # برای پنجره‌های کوتاه و کم‌صدای تشخیص بیدارباش، فیلترهای «سکوت» را شل می‌کنیم
            kwargs.update(
                no_speech_threshold=0.8,
                log_prob_threshold=-2.0,
                compression_ratio_threshold=3.5,
                temperature=0.0,
            )
        segments, _ = self.model.transcribe(self._to_float(audio_int16), **kwargs)
        return "".join(seg.text for seg in segments).strip()
