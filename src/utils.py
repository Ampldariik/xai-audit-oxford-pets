"""Общие утилиты: конфиг, воспроизводимость, выбор устройства, ввод-вывод."""
from __future__ import annotations

import json
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | os.PathLike | None = None) -> dict:
    """Читает YAML-конфиг и превращает относительные пути в абсолютные (от корня репозитория)."""
    path = Path(path) if path else ROOT / "configs" / "default.yaml"
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for k, v in cfg["paths"].items():
        p = Path(v)
        cfg["paths"][k] = p if p.is_absolute() else ROOT / p
    return cfg


def seed_everything(seed: int = 42) -> None:
    """Фиксирует все генераторы случайных чисел — требование воспроизводимости."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # детерминизм cudnn: чуть медленнее, зато повторяемо
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """CUDA → Apple MPS → CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_json(obj, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=_json_default)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


@contextmanager
def timer(name: str, store: dict | None = None):
    """with timer('train'): ... — печатает и (опционально) сохраняет длительность."""
    t0 = time.perf_counter()
    yield
    dt = time.perf_counter() - t0
    print(f"[time] {name}: {dt:.1f} s")
    if store is not None:
        store[name] = dt


def ensure_dirs(cfg: dict) -> None:
    for k in ("cache_dir", "runs_dir", "reports_dir"):
        Path(cfg["paths"][k]).mkdir(parents=True, exist_ok=True)
    (Path(cfg["paths"]["reports_dir"]) / "figures").mkdir(parents=True, exist_ok=True)
