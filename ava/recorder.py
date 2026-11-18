"""تشخیص پایان گفتار کاربر با webrtcvad."""

from __future__ import annotations

import numpy as np
import webrtcvad

from .audio_io import FRAME_MS, Mic


def record_utterance(
    mic: Mic,
    aggressiveness: int = 2,
    silence_ms: int = 800,
    max_seconds: float = 20.0,
    start_timeout: float = 6.0,
) -> np.ndarray:
    """از میکروفون ضبط می‌کند تا وقتی کاربر ساکت شود.

    اگر کاربر اصلاً چیزی نگوید، آرایه‌ی خالی برمی‌گرداند.
    """
    vad = webrtcvad.Vad(aggressiveness)
    preroll = mic.snapshot(0.4)  # کمی از قبل تا ابتدای جمله بریده نشود
    mic.start_capture()
    frames: list[np.ndarray] = [preroll] if preroll.size else []
    voiced = False
    silence_run = 0
    elapsed = 0.0
    waited = 0.0
    try:
        while elapsed < max_seconds * 1000:
            data = mic.read_capture(timeout=1.0)
            if data is None:
                waited += 1.0
                if not voiced and waited >= start_timeout:
                    break
                continue
            frames.append(data)
            elapsed += FRAME_MS
            is_speech = vad.is_speech(data.tobytes(), mic.sample_rate)
            if is_speech:
                voiced = True
                silence_run = 0
            elif voiced:
                silence_run += FRAME_MS
                if silence_run >= silence_ms:
                    break
    finally:
        mic.stop_capture()

    if not voiced or not frames:
        return np.array([], dtype=np.int16)
    return np.concatenate(frames)
