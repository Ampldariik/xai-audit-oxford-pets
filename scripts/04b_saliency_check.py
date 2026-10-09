"""
Неделя 10 (часть 1b). Проверка карт значимости контрольным вмешательством.

Карта значимости полезна, только если она действительно отражает то, на что опирается модель.
Проверяем это двумя независимыми тестами:

1) Тест удаления (deletion / occlusion, Samek et al. 2017; Petsiuk et al. 2018).
   Закрываем 20 % пикселей кадра (заменяем размытым кадром) тремя способами:
     top    — самые «горячие» по Grad-CAM;
     random — случайные пиксели (та же площадь, гладкая случайная маска);
     bottom — самые «холодные».
   Если карта верна: падение вероятности предсказанного класса top ≫ random > bottom.

2) Тест рандомизации модели (Adebayo et al. 2018, «Sanity Checks for Saliency Maps»).
   Заменяем веса layer4 и fc случайными и снова строим Grad-CAM. Если карта
   почти не изменилась (высокая корреляция Спирмена), она отражает не модель, а сам кадр
   (детектор краёв) — такому методу доверять нельзя.

Выход: reports/saliency_check.json, reports/figures/saliency_deletion.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import copy

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr, wilcoxon

from src.data import class_names, load_split_frame, to_tensor_batch
from src.models import gradcam_target_layer, load_checkpoint
from src.utils import get_device, load_config, save_json, seed_everything, timer
from src.xai import gradcam_maps, overlay

cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else None)
seed_everything(cfg["seed"])
P, R = cfg["paths"], cfg["paths"]["reports_dir"]
FIG = R / "figures"
S = cfg["data"]["img_size"]
df = load_split_frame(cfg)
names = class_names(df)
te = np.where(df.split == "test")[0]
imgs = np.load(P["cache_dir"] / f"images_{S}.npy", mmap_mode="r")
device = get_device()
rng = np.random.default_rng(cfg["seed"])

N = min(cfg["xai"].get("n_sanity", 300), len(te))
sel = np.sort(rng.choice(te, N, replace=False))
raw = np.array(imgs[sel])
model, _ = load_checkpoint(P["runs_dir"] / "main" / "model.pt", device)
x = to_tensor_batch(raw).to(device)


@torch.no_grad()
def probs(m, arr_uint8):
    return np.concatenate([torch.softmax(m(to_tensor_batch(arr_uint8[s:s + 64]).to(device)), 1).cpu().numpy()
                           for s in range(0, len(arr_uint8), 64)])


p0 = probs(model, raw)
pred = p0.argmax(1)
with timer("gradcam"):
    cams = gradcam_maps(model, gradcam_target_layer(model), x, pred)

# ---------- 1. тест удаления ----------
FRAC = 0.20


def occlude(img, keep_mask):
    """Пиксели вне keep_mask заменяем сильно размытым кадром (убираем информацию, не добавляя новых краёв)."""
    blur = cv2.GaussianBlur(img, (0, 0), 15)
    m = keep_mask[..., None].astype(np.float32)
    return (img * m + blur * (1 - m)).astype(np.uint8)


variants = {"top": [], "random": [], "bottom": []}
for k in range(N):
    c = cams[k] + 1e-6 * rng.random(cams[k].shape)  # разрыв «ничьих» в плоских областях
    thr_top, thr_bot = np.quantile(c, 1 - FRAC), np.quantile(c, FRAC)
    # гладкая случайная карта той же природы (низкочастотный шум), чтобы форма «дыр» была сопоставима
    noise = cv2.GaussianBlur(rng.random((S, S)).astype(np.float32), (0, 0), 12)
    thr_rnd = np.quantile(noise, 1 - FRAC)
    variants["top"].append(occlude(raw[k], c < thr_top))
    variants["bottom"].append(occlude(raw[k], c > thr_bot))
    variants["random"].append(occlude(raw[k], noise < thr_rnd))

res = {"n_images": int(N), "occluded_fraction": FRAC, "deletion": {}}
drops = {}
for v, arr in variants.items():
    pv = probs(model, np.stack(arr))
    drop = p0[np.arange(N), pred] - pv[np.arange(N), pred]
    drops[v] = drop
    res["deletion"][v] = dict(mean_drop_p_pred=float(drop.mean()), median_drop=float(np.median(drop)),
                              pred_changed=float((pv.argmax(1) != pred).mean()))
res["deletion"]["wilcoxon_p_top_vs_random"] = float(wilcoxon(drops["top"], drops["random"]).pvalue)
res["deletion"]["wilcoxon_p_random_vs_bottom"] = float(wilcoxon(drops["random"], drops["bottom"]).pvalue)
res["deletion"]["passed"] = bool(drops["top"].mean() > drops["random"].mean() > drops["bottom"].mean())

# ---------- 2. тест рандомизации модели ----------
rand_model = copy.deepcopy(model)
torch.manual_seed(0)
for mod in list(rand_model.layer4.modules()) + [rand_model.fc]:
    if hasattr(mod, "reset_parameters"):
        mod.reset_parameters()
rand_model.eval()
with timer("gradcam_randomized"):
    cams_rand = gradcam_maps(rand_model, gradcam_target_layer(rand_model), x, pred)
rho = np.array([spearmanr(cams[k].ravel(), cams_rand[k].ravel()).correlation for k in range(N)])
rho = np.nan_to_num(rho)
res["model_randomization"] = dict(randomized="layer4 + fc", spearman_mean=float(rho.mean()),
                                  spearman_median=float(np.median(rho)), share_rho_above_0_5=float((rho > 0.5).mean()),
                                  passed=bool(np.median(rho) < 0.5))
save_json(res, R / "saliency_check.json")
print(res)

# ---------- рисунок ----------
fig = plt.figure(figsize=(17, 7.5))
gs = fig.add_gridspec(3, 6, width_ratios=[1, 1, 1, 1, 1, 1.6])
for r in range(3):
    k = r * 7
    row = [(raw[k], f"{names[pred[k]]} p={p0[k, pred[k]]:.2f}"), (overlay(raw[k], cams[k]), "Grad-CAM"),
           (variants["top"][k], "закрыты топ-20%"), (variants["random"][k], "закрыты случайные 20%"),
           (overlay(raw[k], cams_rand[k]), f"Grad-CAM рандом. модели\nρ={rho[k]:.2f}")]
    for c, (im, t) in enumerate(row):
        a = fig.add_subplot(gs[r, c]); a.imshow(im); a.set_title(t, fontsize=8); a.axis("off")
ax = fig.add_subplot(gs[:, 5])
lab = {"top": "топ-20% Grad-CAM", "random": "случайные 20%", "bottom": "нижние 20%"}
vals = [res["deletion"][v]["mean_drop_p_pred"] for v in lab]
ax.bar(list(lab.values()), vals, color=["#C44E52", "#8C8C8C", "#4C72B0"])
for i, v in enumerate(vals):
    ax.text(i, v, f"{v:.3f}", ha="center", va="bottom")
ax.set_ylabel("Среднее падение p(предсказанного класса)")
ax.set_title(f"Тест удаления, N={N}")
ax.tick_params(axis="x", rotation=12)
fig.suptitle("Контрольное вмешательство: проверка достоверности карт Grad-CAM")
plt.tight_layout()
plt.savefig(FIG / "saliency_deletion.png", dpi=120)
print("Готово.")
