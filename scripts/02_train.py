"""
Неделя 5. Обучение моделей.

  python scripts/02_train.py color_logreg   # baseline-0: логистическая регрессия на цветовых гистограммах
  python scripts/02_train.py baseline_cnn   # baseline-1: маленькая CNN с нуля
  python scripts/02_train.py main           # основная модель: ResNet (ImageNet) + fine-tuning

Выбор эпохи — только по val Macro-F1. Тестовая выборка здесь не используется
(кроме color_logreg, для которого сразу сохраняем вероятности на всех сплитах).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import time

import cv2
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data import PetsCache, class_names, load_split_frame
from src.models import build_model
from src.train import fit
from src.utils import get_device, load_config, save_json, seed_everything

kind = sys.argv[1] if len(sys.argv) > 1 else "main"
cfg = load_config(sys.argv[2] if len(sys.argv) > 2 else None)
seed_everything(cfg["seed"])
P = cfg["paths"]
df = load_split_frame(cfg)
names = class_names(df)
S = cfg["data"]["img_size"]
out = P["runs_dir"] / kind
out.mkdir(parents=True, exist_ok=True)
idx = {s: np.where(df.split == s)[0] for s in ("train", "val", "test")}
y = df["label"].values

if kind == "color_logreg":
    # ---- Baseline-0: «модель, которая видит только цвета» --------------------
    # Если она заметно лучше случайного угадывания (1/37 ≈ 2.7 %), значит часть сигнала
    # содержится в цвете шерсти и ФОНА — полезная отправная точка для аудита.
    imgs = np.load(P["cache_dir"] / f"images_{S}.npy", mmap_mode="r")

    def feats(i):
        hsv = cv2.cvtColor(np.ascontiguousarray(imgs[i]), cv2.COLOR_RGB2HSV)
        h2 = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256]).ravel()
        v = cv2.calcHist([hsv], [2], None, [16], [0, 256]).ravel()
        f = np.concatenate([h2, v])
        return f / f.sum()

    t0 = time.perf_counter()
    X = np.stack([feats(i) for i in range(len(df))])
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=3000))
    clf.fit(X[idx["train"]], y[idx["train"]])
    probs = clf.predict_proba(X)
    np.save(out / "probs_all.npy", probs)
    hist = dict(val_macro_f1=f1_score(y[idx["val"]], probs[idx["val"]].argmax(1), average="macro"),
                train_time_s=time.perf_counter() - t0)
    save_json(hist, out / "history.json")
    print(hist)
    sys.exit()

# ---- нейросети -----------------------------------------------------------
device = get_device()
print("Устройство:", device)
tcfg = cfg["baseline_cnn"] if kind == "baseline_cnn" else cfg["main_model"]
size = tcfg.get("img_size", S)
model_kind = "baseline_cnn" if kind == "baseline_cnn" else "main"
model = build_model(model_kind, len(names), cfg, pretrained=True)
n_params = sum(p.numel() for p in model.parameters())
n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Параметров: {n_params / 1e6:.2f} M (обучаемых {n_train / 1e6:.2f} M)")

train_ds = PetsCache(P["cache_dir"], idx["train"], y[idx["train"]], size, train=True, cache_size=S)
val_ds = PetsCache(P["cache_dir"], idx["val"], y[idx["val"]], size, train=False, cache_size=S)

t0 = time.perf_counter()
model, history = fit(model, model_kind, train_ds, val_ds, tcfg, device, out, num_workers=cfg["data"]["num_workers"])
train_time = time.perf_counter() - t0
torch.save(dict(kind=model_kind, num_classes=len(names), cfg=cfg,
                state_dict=model.state_dict(), img_size=size, class_names=names), out / "model.pt")
save_json(dict(history=history, train_time_s=train_time, n_params=n_params, n_trainable=n_train,
               device=str(device), img_size=size), out / "history.json")
print(f"Сохранено: {out / 'model.pt'} ({train_time / 60:.1f} мин)")
