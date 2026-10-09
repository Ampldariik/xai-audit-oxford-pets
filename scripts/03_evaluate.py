"""
Неделя 5. Оценка baseline и основной модели на ОТЛОЖЕННОМ тесте.

Выход: reports/metrics_*.json, reports/model_comparison.csv,
       reports/predictions_main.csv (вход для атласа ошибок),
       reports/figures/confusion_*.png, reports/figures/training_curves.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from src.data import PetsCache, class_names, load_split_frame
from src.metrics import full_report, plot_confusion, species_accuracy, top_confusions
from src.models import load_checkpoint
from src.train import predict_proba
from src.utils import get_device, load_config, load_json, save_json

cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else None)
P, R = cfg["paths"], cfg["paths"]["reports_dir"]
FIG = R / "figures"
df = load_split_frame(cfg)
names = class_names(df)
species_of_label = df.drop_duplicates("label").sort_values("label")["species"].values
te = np.where(df.split == "test")[0]
y = df.label.values[te]
device = get_device()
S = cfg["data"]["img_size"]

pretty = {"color_logreg": "Baseline-0: LogReg на цветовых гистограммах",
          "baseline_cnn": "Baseline-1: CNN с нуля (4 блока)",
          "linear_probe": f"Linear probe: LogReg на замороженных признаках {cfg['main_model']['arch']}",
          "main": f"Основная: {cfg['main_model']['arch']} (ImageNet) + fine-tuning"}
rows = []
for kind in ["color_logreg", "baseline_cnn", "linear_probe", "main"]:
    run = P["runs_dir"] / kind
    if not run.exists():
        print("нет", run)
        continue
    if kind in ("color_logreg", "linear_probe"):
        probs = np.load(run / "probs_all.npy")[te]
    else:
        model, ck = load_checkpoint(run / "model.pt", device)
        ds = PetsCache(P["cache_dir"], te, y, ck["img_size"], train=False, cache_size=S)
        probs, _ = predict_proba(model, DataLoader(ds, batch_size=64, num_workers=cfg["data"]["num_workers"]), device)
    pred = probs.argmax(1)
    rep = full_report(y, probs, names)
    rep["species_accuracy"] = species_accuracy(y, pred, species_of_label)
    save_json(rep, R / f"metrics_{kind}.json")
    hist = load_json(run / "history.json")
    rows.append(dict(model=pretty[kind], accuracy=rep["accuracy"], macro_f1=rep["macro_f1"],
                     macro_f1_ci95=f"[{rep['macro_f1_ci95'][0]:.3f}; {rep['macro_f1_ci95'][1]:.3f}]",
                     macro_precision=rep["macro_precision"], macro_recall=rep["macro_recall"],
                     top5_accuracy=rep["top5_accuracy"], species_accuracy=rep["species_accuracy"],
                     train_time_min=hist.get("train_time_s", 0) / 60, n_params_M=hist.get("n_params", 0) / 1e6))
    print(f"{kind:13s} acc={rep['accuracy']:.4f} macroF1={rep['macro_f1']:.4f} {rep['macro_f1_ci95']}")
    if kind not in ("color_logreg", "linear_probe"):
        plot_confusion(y, pred, names, FIG / f"confusion_{kind}.png",
                       f"{pretty[kind]}: нормированная матрица ошибок (test, N={len(y)})")
    if kind == "main":
        top = np.argsort(-probs, 1)
        out = df.iloc[te][["image_id", "label", "class_name", "species", "group", "sharpness", "brightness",
                           "contrast", "fg_frac", "orig_w", "orig_h"]].copy()
        out["cache_idx"] = te
        out["pred"] = pred
        out["pred_name"] = [names[p] for p in pred]
        out["confidence"] = probs.max(1)
        out["p_true"] = probs[np.arange(len(y)), y]
        out["top2"] = top[:, 1]
        out["top2_name"] = [names[p] for p in top[:, 1]]
        out["margin"] = probs[np.arange(len(y)), top[:, 0]] - probs[np.arange(len(y)), top[:, 1]]
        out["true_in_top3"] = (top[:, :3] == y[:, None]).any(1)
        out["correct"] = pred == y
        out.to_csv(R / "predictions_main.csv", index=False)
        np.save(R / "probs_main_test.npy", probs)
        top_confusions(y, pred, names).to_csv(R / "top_confusions_main.csv", index=False)

pd.DataFrame(rows).to_csv(R / "model_comparison.csv", index=False)
print(pd.DataFrame(rows).to_string())

# кривые обучения
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for kind, c in [("baseline_cnn", "#DD8452"), ("main", "#4C72B0")]:
    f = P["runs_dir"] / kind / "history.json"
    if not f.exists():
        continue
    h = pd.DataFrame(load_json(f)["history"])
    axes[0].plot(h.epoch, h.train_loss, "-o", c=c, label=kind)
    axes[1].plot(h.epoch, h.val_macro_f1, "-o", c=c, label=kind)
axes[0].set_title("Train loss"); axes[1].set_title("Validation Macro-F1")
for a in axes:
    a.set_xlabel("Эпоха"); a.legend(); a.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(FIG / "training_curves.png", dpi=140)
