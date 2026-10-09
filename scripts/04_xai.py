"""
Неделя 10 (часть 1). Карты значимости: Grad-CAM для всего теста + Integrated Gradients на подвыборке.

Количественно проверяем, «смотрит» ли сеть на животное: FG-energy, pointing game, FG-lift
(см. src/xai.py). Сравниваем основную модель и baseline CNN, верные и ошибочные предсказания.

Выход: reports/xai_attention_{main,baseline_cnn}.csv, reports/xai_summary.json,
       reports/figures/xai_gradcam_examples.png, reports/figures/xai_gradcam_vs_ig.png,
       runs/xai/cams_main.npy (используется атласом ошибок)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from src.data import class_names, load_split_frame, to_tensor_batch
from src.models import gradcam_target_layer, load_checkpoint
from src.utils import get_device, load_config, save_json, seed_everything, timer
from src.xai import attention_metrics, gradcam_maps, integrated_gradients, overlay

cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else None)
seed_everything(cfg["seed"])
P, R = cfg["paths"], cfg["paths"]["reports_dir"]
FIG = R / "figures"
S = cfg["data"]["img_size"]
df = load_split_frame(cfg)
names = class_names(df)
te = np.where(df.split == "test")[0]
y = df.label.values[te]
imgs = np.load(P["cache_dir"] / f"images_{S}.npy", mmap_mode="r")
msks = np.load(P["cache_dir"] / f"masks_{S}.npy", mmap_mode="r")
device = get_device()
(P["runs_dir"] / "xai").mkdir(parents=True, exist_ok=True)

summary = {}
cams_main = None
for kind in ["main", "baseline_cnn"]:
    model, ck = load_checkpoint(P["runs_dir"] / kind / "model.pt", device)
    size = ck["img_size"]
    x = to_tensor_batch(imgs[te], size)
    with timer(f"gradcam_{kind}"):
        import torch

        with torch.no_grad():
            pred = np.concatenate([model(x[s:s + 128].to(device)).argmax(1).cpu().numpy() for s in range(0, len(x), 128)])
        cams = gradcam_maps(model, gradcam_target_layer(model), x.to(device), pred)
    rows = []
    for k, i in enumerate(te):
        tm = msks[i] if size == S else cv2.resize(msks[i], (size, size), interpolation=cv2.INTER_NEAREST)
        rows.append(dict(image_id=df.image_id[i], cache_idx=i, label=y[k], pred=pred[k], correct=pred[k] == y[k],
                         **attention_metrics(cams[k], tm)))
    att = pd.DataFrame(rows)
    att.to_csv(R / f"xai_attention_{kind}.csv", index=False)
    c, w = att[att.correct], att[~att.correct]
    p_mw = mannwhitneyu(c.fg_energy, w.fg_energy).pvalue if len(w) > 1 else None
    summary[kind] = dict(
        fg_energy_mean=att.fg_energy.mean(), fg_frac_mean=att.fg_frac.mean(), fg_lift_mean=att.fg_lift.mean(),
        pointing_game=att.pointing_hit.mean(),
        fg_energy_correct=c.fg_energy.mean(), fg_energy_wrong=w.fg_energy.mean() if len(w) else None,
        pointing_correct=c.pointing_hit.mean(), pointing_wrong=w.pointing_hit.mean() if len(w) else None,
        share_fg_energy_below_0_5=(att.fg_energy < 0.5).mean(),
        mannwhitney_p_fg_energy_correct_vs_wrong=p_mw, n_correct=len(c), n_wrong=len(w),
    )
    print(kind, summary[kind])
    if kind == "main":
        cams_main, pred_main, att_main = cams, pred, att
        np.save(P["runs_dir"] / "xai" / "cams_main.npy", cams.astype(np.float16))

# ---- Рисунок 1: Grad-CAM на верных и ошибочных примерах --------------------------------
rng = np.random.default_rng(1)
ok = rng.choice(np.where(att_main.correct)[0], 6, replace=False)
bad = rng.choice(np.where(~att_main.correct)[0], min(6, (~att_main.correct).sum()), replace=False)
fig, axes = plt.subplots(2, 6, figsize=(18, 6.6))
for row, sel, tag in [(0, ok, "ВЕРНО"), (1, bad, "ОШИБКА")]:
    for a, k in zip(axes[row], sel):
        i = te[k]
        a.imshow(overlay(imgs[i], cams_main[k]))
        a.set_title(f"{tag}: {names[y[k]]}\n→ {names[pred_main[k]]}  FG={att_main.fg_energy[k]:.2f}", fontsize=8)
        a.contour((msks[i] != 2).astype(float), levels=[0.5], colors="white", linewidths=0.7)
for a in axes.ravel():
    a.axis("off")
fig.suptitle("Grad-CAM основной модели (белый контур — маска животного, FG — доля внимания на животном)")
plt.tight_layout()
plt.savefig(FIG / "xai_gradcam_examples.png", dpi=130)
plt.close()

# ---- Рисунок 2: Grad-CAM vs Integrated Gradients ------------------------------------------
model, ck = load_checkpoint(P["runs_dir"] / "main" / "model.pt", device)
n_ig = cfg["xai"]["n_examples_ig"]
sel = np.concatenate([rng.choice(np.where(att_main.correct)[0], n_ig // 2, replace=False),
                      rng.choice(np.where(~att_main.correct)[0], min(n_ig // 2, (~att_main.correct).sum()), replace=False)])
with timer("integrated_gradients"):
    ig = integrated_gradients(model, to_tensor_batch(imgs[te[sel]]).to(device), pred_main[sel], cfg["xai"]["ig_steps"])
ig_fg = [attention_metrics(cv2.GaussianBlur(ig[k], (0, 0), 3), msks[te[s]])["fg_energy"] for k, s in enumerate(sel)]
summary["integrated_gradients"] = dict(n=len(sel), fg_energy_mean=float(np.mean(ig_fg)),
                                       gradcam_fg_energy_same_images=float(att_main.fg_energy.values[sel].mean()),
                                       spearman_gradcam_vs_ig=float(pd.Series(ig_fg).corr(
                                           pd.Series(att_main.fg_energy.values[sel]), method="spearman")))
show = sel[:4].tolist() + sel[n_ig // 2: n_ig // 2 + 4].tolist()
fig, axes = plt.subplots(3, 8, figsize=(20, 8))
for c, s in enumerate(show):
    k = list(sel).index(s)
    i = te[s]
    axes[0, c].imshow(imgs[i]); axes[0, c].set_title(f"{names[y[s]]}\n→ {names[pred_main[s]]}", fontsize=8)
    axes[1, c].imshow(overlay(imgs[i], cams_main[s]))
    axes[2, c].imshow(ig[k], cmap="inferno")
for a in axes.ravel():
    a.axis("off")
axes[1, 0].set_ylabel("Grad-CAM"); axes[2, 0].set_ylabel("IG")
fig.suptitle("Сверху: кадр; посередине: Grad-CAM; снизу: Integrated Gradients (слева 4 верных, справа 4 ошибки)")
plt.tight_layout()
plt.savefig(FIG / "xai_gradcam_vs_ig.png", dpi=120)
plt.close()

# ---- Рисунок 3: распределение FG-energy ---------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(att_main.fg_energy[att_main.correct], bins=30, alpha=0.6, density=True, label="верные")
ax.hist(att_main.fg_energy[~att_main.correct], bins=30, alpha=0.6, density=True, label="ошибки")
ax.axvline(att_main.fg_frac.mean(), color="k", ls="--", label="ср. доля площади животного")
ax.set_xlabel("FG-energy (доля Grad-CAM на животном)"); ax.legend(); ax.set_title("Основная модель, test")
plt.tight_layout()
plt.savefig(FIG / "xai_fg_energy_hist.png", dpi=130)
save_json(summary, R / "xai_summary.json")
print("Готово.")
