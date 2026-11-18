"""بارگذاری پیکربندی از فایل YAML با مقادیر پیش‌فرض."""

from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import yaml

DEFAULTS: dict = {
    "language": "fa",
    "interaction": "push_to_talk",  # push_to_talk | wake
    "terminal_reshape": True,  # نمایش راست‌به‌چپِ فارسی برای کنسول ویندوز؛ در ترمینال‌های مدرن false کن
    "debug": False,
    "wake": {
        "words": ["آوا", "اوا", "ava"],
        "model": "base",
        "window_seconds": 2.0,
        "hop_seconds": 1.0,
        "rms_gate": 1200,
        "debug": False,
    },
    "stt": {"model": "small", "device": "cpu", "compute_type": "int8"},
    "tts": {
        "model_path": "models/piper/fa_IR-amir-medium.onnx",
        "length_scale": 1.05,
        "robot": False,
        "robot_carrier_hz": 26.0,
        "robot_mix": 0.12,
        "robot_bits": 8,
        "sound_effects": True,
    },
    "llm": {
        "host": "http://localhost:11434",
        "model": "qwen2.5:7b",
        "temperature": 0.5,
        "num_ctx": 3072,
        "num_predict": 140,
        "history_turns": 4,
        "timeout": 600,
        "keep_alive": "30m",
    },
    "websearch": {"enabled": True, "max_results": 5, "region": "wt-wt"},
    "memory": {"path": "memory/ava_memory.json"},
    "audio": {
        "sample_rate": 16000,
        "input_device": None,
        "output_device": None,
        "input_gain": 1.0,
    },
    "vad": {"aggressiveness": 2, "silence_ms": 800, "max_utterance_seconds": 20},
}


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _to_namespace(obj):
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in obj.items()})
    return obj


def load_config(path: str | Path = "config.yaml") -> SimpleNamespace:
    path = Path(path)
    user_cfg: dict = {}
    if path.exists():
        user_cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    merged = _deep_merge(DEFAULTS, user_cfg)
    cfg = _to_namespace(merged)
    cfg.project_root = str(path.resolve().parent)
    return cfg
