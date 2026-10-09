"""
Неделя 5 (дополнительно). Scikit-learn поверх «замороженных» признаков ImageNet (linear probe).

Промежуточная точка сравнения между baseline и полным fine-tuning:
ResNet без дообучения → 512-мерный вектор (выход global average pooling) →
LogisticRegression (scikit-learn), C подбирается по Macro-F1 на validation.

Отвечает на вопрос: сколько качества даёт сам перенос признаков ImageNet,
а сколько — дообучение сети на наших породах.

Выход: runs/linear_probe/probs_all.npy, runs/linear_probe/history.json
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import time

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data import load_split_frame, to_tensor_batch
from src.models import build_main_model
from src.utils import get_device, load_config, save_json, seed_everything

cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else None)
seed_everything(cfg["seed"])
P = cfg["paths"]
S = cfg["data"]["img_size"]
df = load_split_frame(cfg)
y = df.label.values
idx = {s: np.where(df.split == s)[0] for s in ("train", "val", "test")}
out = P["runs_dir"] / "linear_probe"
out.mkdir(parents=True, exist_ok=True)
device = get_device()

t0 = time.perf_counter()
feat_file = out / "features.npy"
if feat_file.exists():
    X = np.load(feat_file)
else:
    net = build_main_model(cfg["main_model"]["arch"], 1000, pretrained=True)
    net.fc = torch.nn.Identity()  # выход = признаки после global average pooling
    net.to(device).eval()
    imgs = np.load(P["cache_dir"] / f"images_{S}.npy", mmap_mode="r")
    feats = []
    with torch.no_grad():
        for s in range(0, len(df), 128):
            feats.append(net(to_tensor_batch(imgs[s:s + 128]).to(device)).cpu().numpy())
            if (s // 128) % 10 == 0:
                print(f"  признаки {s}/{len(df)}", flush=True)
    X = np.concatenate(feats).astype(np.float32)
    np.save(feat_file, X)
t_feat = time.perf_counter() - t0

best = None
for C in [0.01, 0.03, 0.1, 0.3, 1.0]:
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))
    clf.fit(X[idx["train"]], y[idx["train"]])
    f1 = f1_score(y[idx["val"]], clf.predict(X[idx["val"]]), average="macro")
    print(f"C={C}: val Macro-F1={f1:.4f}")
    if best is None or f1 > best[0]:
        best = (f1, C, clf)
f1, C, clf = best
np.save(out / "probs_all.npy", clf.predict_proba(X))
save_json(dict(val_macro_f1=f1, best_C=C, feature_time_s=t_feat, train_time_s=time.perf_counter() - t0,
               n_params=X.shape[1] * 37 + 37), out / "history.json")
print(f"Лучший C={C}, val Macro-F1={f1:.4f}")
