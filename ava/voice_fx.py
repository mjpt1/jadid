"""افکت صوتی رباتیک برای صدای آوا + بلیپ‌های رابط.

همه با numpy انجام می‌شود؛ وابستگی اضافه ندارد.
"""

from __future__ import annotations

import numpy as np


def _normalize(x: np.ndarray, headroom: float = 0.95) -> np.ndarray:
    peak = float(np.max(np.abs(x))) or 1.0
    return x / peak * headroom


def _lowpass(x: np.ndarray, cutoff_hz: float, sample_rate: int) -> np.ndarray:
    """پایین‌گذرِ سریع (میانگین متحرکِ آبشاری) برای بم‌تر کردن."""
    length = max(2, int(round(sample_rate / max(cutoff_hz, 1.0))))
    kernel = np.ones(length, dtype=np.float32) / length
    y = x
    for _ in range(2):  # دو بار = شیب تندتر
        y = np.convolve(y, kernel, mode="same")
    return y.astype(np.float32)


def robotize(
    audio_int16: np.ndarray,
    sample_rate: int,
    carrier_hz: float = 28.0,
    dry_mix: float = 0.12,
    bits: int = 8,
) -> np.ndarray:
    """لحن رباتیکِ بم و فلزی: مدولاسیون حلقوی + شانه‌ایِ چندضربه + کاهش بیت + پایین‌گذر."""
    if audio_int16 is None or audio_int16.size == 0:
        return audio_int16

    x = audio_int16.astype(np.float32) / 32768.0
    n = x.shape[0]
    t = np.arange(n, dtype=np.float32) / sample_rate

    # ۱) مدولاسیون حلقویِ دوتایی (فرکانس پایین‌تر = بم‌تر)
    c1 = np.sign(np.sin(2 * np.pi * carrier_hz * t)).astype(np.float32)
    c2 = np.sin(2 * np.pi * (carrier_hz * 1.5) * t).astype(np.float32)
    wet = x * (0.75 * c1 + 0.25 * c2)

    # ۲) فیلتر شانه‌ایِ چندضربه برای طنین فلزیِ قوی
    combed = wet.copy()
    for ms, fb in ((5.0, 0.6), (11.0, 0.4), (17.0, 0.25)):
        d = max(1, int(sample_rate * ms / 1000))
        combed[d:] += fb * wet[:-d]

    # ۳) کاهش عمق بیت (حس دیجیتال)
    levels = float(2 ** max(4, bits))
    crushed = np.round(combed * levels) / levels

    # ۴) پایین‌گذر برای بم‌تر شدن
    bassy = _lowpass(crushed, 2200.0, sample_rate)

    out = _normalize(bassy) * (1.0 - dry_mix) + _normalize(x) * dry_mix
    return (_normalize(out, 0.97) * 32767).astype(np.int16)


def blip(
    sample_rate: int,
    start_hz: float = 420.0,
    end_hz: float = 900.0,
    ms: int = 160,
    robotic: bool = True,
) -> np.ndarray:
    """یک بلیپ کوتاهِ فرکانس‌متغیر برای نشانه‌های رابط."""
    n = int(sample_rate * ms / 1000)
    t = np.arange(n, dtype=np.float32) / sample_rate
    sweep = np.linspace(start_hz, end_hz, n, dtype=np.float32)
    phase = 2 * np.pi * np.cumsum(sweep) / sample_rate
    tone = np.sin(phase).astype(np.float32)
    if robotic:
        tone = np.sign(tone) * 0.6 + tone * 0.4  # کمی مربعی
    env = np.ones(n, dtype=np.float32)
    fade = max(1, int(0.02 * sample_rate))
    env[:fade] = np.linspace(0, 1, fade)
    env[-fade:] = np.linspace(1, 0, fade)
    return (_normalize(tone * env) * 0.4 * 32767).astype(np.int16)


def listen_cue(sample_rate: int) -> np.ndarray:
    """نشانه‌ی «گوش می‌دهم» — بلیپ بالارونده."""
    return blip(sample_rate, 400, 950, 150)


def thinking_cue(sample_rate: int) -> np.ndarray:
    """نشانه‌ی «در حال پردازش» — دو بلیپ کوتاه."""
    a = blip(sample_rate, 700, 700, 70)
    gap = np.zeros(int(sample_rate * 0.05), dtype=np.int16)
    b = blip(sample_rate, 700, 500, 70)
    return np.concatenate([a, gap, b])


def error_cue(sample_rate: int) -> np.ndarray:
    """نشانه‌ی «متوجه نشدم» — بلیپ پایین‌رونده."""
    return blip(sample_rate, 700, 300, 200)
