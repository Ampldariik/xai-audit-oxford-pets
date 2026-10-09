"""
Неделя 10 (часть 2). Эксперимент с контролируемой подменой фона.

Для каждого тестового кадра строим 6 вариантов (src/background.py) и сравниваем
предсказания «до/после»: Accuracy, Macro-F1, средняя вероятность истинного класса,
ΔConfidence (парная разность), доля смен предсказания (flip rate), смена вида кошка↔собака.
Значимость Δ проверяем парным критерием Уилкоксона.

Выход: reports/bg_swap_metrics.csv, reports/bg_swap_per_image.csv, reports/bg_swap_per_class.csv,
       reports/figures/bg_swap_examples.png, reports/figures/bg_swap_bars.png, reports/figures/bg_swap_per_class.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon
from sklearn.metrics import f1_score

from src.background import VARIANTS, build_variants
from src.data import class_names, load_split_frame, to_tensor_batch
from src.models import load_checkpoint
from src.utils import get_device, load_config, save_json, seed_everything, timer

cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else None)
seed_everything(cfg["seed"])
P, R = cfg["paths"], cfg["paths"]["reports_dir"]
FIG = R / "figures"
S = cfg["data"]["img_size"]
df = load_split_frame(cfg)
names = class_names(df)
species_of_label = df.drop_duplicates("label").sort_values("label")["species"].values
te = np.where(df.split == "test")[0]
y = df.label.values[te]
imgs = np.load(P["cache_dir"] / f"images_{S}.npy", mmap_mode="r")
msks = np.load(P["cache_dir"] / f"masks_{S}.npy", mmap_mode="r")
device = get_device()

# фоны берём из ВАЛИДАЦИИ, чтобы в тестовый кадр не попал объект другого тестового кадра
rng = np.random.default_rng(cfg["seed"])
pool = rng.choice(np.where(df.split == "val")[0], cfg["background_swap"]["n_bg_pool"], replace=False)
with timer("build_variants"):
    V, info = build_variants(imgs, msks, te, df.species.values, pool, cfg["background_swap"]["mask_dilate_px"], cfg["seed"])


@torch.no_grad()
def probs_of(model, arr, size):
    out = []
    for s in range(0, len(arr), 64):
        out.append(torch.softmax(model(to_tensor_batch(arr[s:s + 64], size).to(device)), 1).cpu().numpy())
    return np.concatenate(out)


rows, per_image = [], {"image_id": df.image_id.values[te], "label": y, "cache_idx": te}
for kind in ["main", "baseline_cnn"]:
    model, ck = load_checkpoint(P["runs_dir"] / kind / "model.pt", device)
    Pv = {v: probs_of(model, V[v], ck["img_size"]) for v in VARIANTS}
    p0 = Pv["original"]
    pred0 = p0.argmax(1)
    pt0 = p0[np.arange(len(y)), y]
    for v in VARIANTS:
        p = Pv[v]
        pred = p.argmax(1)
        pt = p[np.arange(len(y)), y]
        d = pt - pt0
        rows.append(dict(
            model=kind, variant=v, accuracy=(pred == y).mean(), macro_f1=f1_score(y, pred, average="macro"),
            mean_p_true=pt.mean(), mean_conf_top1=p.max(1).mean(),
            delta_p_true_mean=d.mean(), delta_p_true_median=np.median(d),
            flip_rate=(pred != pred0).mean(),
            species_flip_rate=(species_of_label[pred] != species_of_label[pred0]).mean(),
            species_accuracy=(species_of_label[pred] == species_of_label[y]).mean(),
            wilcoxon_p=(wilcoxon(pt, pt0).pvalue if v != "original" else None),
        ))
        if kind == "main":
            per_image[f"p_true_{v}"] = pt
            per_image[f"pred_{v}"] = pred
    if kind == "main":
        Pmain = Pv

M = pd.DataFrame(rows)
M.to_csv(R / "bg_swap_metrics.csv", index=False)
print(M.round(4).to_string())
pi = pd.DataFrame(per_image)
pi["bg_random_src"] = info["bg_random_src"]
pi["bg_species_src"] = info["bg_species_src"]
pi.to_csv(R / "bg_swap_per_image.csv", index=False)

# по классам: насколько падает уверенность при чужом фоне
pi["class_name"] = [names[l] for l in y]
pc = pi.groupby("class_name").agg(
    acc_original=("pred_original", lambda s: (s.values == y[s.index]).mean()),
    acc_bg_swap_random=("pred_bg_swap_random", lambda s: (s.values == y[s.index]).mean()),
    acc_fg_gray=("pred_fg_gray", lambda s: (s.values == y[s.index]).mean()),
    d_conf_swap=("p_true_bg_swap_random", "mean"),
    p_true_original=("p_true_original", "mean"),
    p_true_bg_only=("p_true_bg_only", "mean"),
).reset_index()
pc["d_conf_swap"] = pc["d_conf_swap"] - pc["p_true_original"]
pc = pc.sort_values("d_conf_swap")
pc.to_csv(R / "bg_swap_per_class.csv", index=False)

# ---- рисунки ----------------------------------------------------------------------
ex = rng.choice(len(te), 5, replace=False)
fig, axes = plt.subplots(len(ex), len(VARIANTS), figsize=(3 * len(VARIANTS), 3.2 * len(ex)))
for r, k in enumerate(ex):
    for c, v in enumerate(VARIANTS):
        a = axes[r, c]
        a.imshow(V[v][k])
        pr = Pmain[v][k]
        a.set_title(f"{v}\n{names[pr.argmax()]} ({pr.max():.2f}) | p_true={pr[y[k]]:.2f}", fontsize=7,
                    color="green" if pr.argmax() == y[k] else "red")
        a.axis("off")
fig.suptitle("Подмена фона: предсказание основной модели (зелёный — верно, красный — ошибка)")
plt.tight_layout()
plt.savefig(FIG / "bg_swap_examples.png", dpi=110)
plt.close()

fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
w = 0.38
xs = np.arange(len(VARIANTS))
for j, (kind, col) in enumerate([("baseline_cnn", "#DD8452"), ("main", "#4C72B0")]):
    m = M[M.model == kind].set_index("variant").loc[VARIANTS]
    axes[0].bar(xs + (j - 0.5) * w, m.accuracy, w, label=kind, color=col)
    axes[1].bar(xs + (j - 0.5) * w, m.mean_p_true, w, label=kind, color=col)
for a, t in zip(axes, ["Accuracy", "Средняя вероятность истинного класса"]):
    a.set_xticks(xs, VARIANTS, rotation=20)
    a.set_title(t)
    a.axhline(1 / 37, color="gray", ls=":", lw=1)
    a.legend()
    a.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(FIG / "bg_swap_bars.png", dpi=130)
plt.close()

fig, ax = plt.subplots(figsize=(10, 9))
ax.barh(pc.class_name, pc.d_conf_swap, color=np.where(pc.d_conf_swap < pc.d_conf_swap.median(), "#C44E52", "#8C8C8C"))
ax.set_xlabel("Δ p(истинный класс): bg_swap_random − original")
ax.set_title("Какие породы сильнее всего зависят от фона (основная модель)")
plt.tight_layout()
plt.savefig(FIG / "bg_swap_per_class.png", dpi=130)
print("Готово.")
