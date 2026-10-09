"""
Неделя 10 (часть 3). Положительный контроль: подсаженный водяной знак (planted shortcut).

1) На poison_frac train-кадров целевого класса добавляем надпись в правый нижний угол.
2) Дообучаем такую же ResNet («отравленная» модель).
3) Сравниваем отравленную и чистую (main) модели:
   * точность на чистом тесте;
   * attack rate: надпись на кадрах других классов → доля предсказаний «целевой класс»;
   * recall целевого класса без надписи и с надписью;
   * доля Grad-CAM в прямоугольнике надписи.

Выход: reports/watermark_results.json, reports/figures/watermark_gradcam.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import f1_score

from src.data import PetsCache, class_names, load_split_frame, to_tensor_batch
from src.models import build_model, gradcam_target_layer, load_checkpoint
from src.train import fit
from src.utils import get_device, load_config, save_json, seed_everything, timer
from src.watermark import Stamp, add_watermark, box_energy, watermark_box
from src.xai import gradcam_maps, overlay

cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else None)
seed_everything(cfg["seed"])
P, R = cfg["paths"], cfg["paths"]["reports_dir"]
FIG = R / "figures"
S = cfg["data"]["img_size"]
W = cfg["watermark"]
df = load_split_frame(cfg)
names = class_names(df)
tgt = names.index(W["target_class"])
y_all = df.label.values
idx = {s: np.where(df.split == s)[0] for s in ("train", "val", "test")}
device = get_device()
TAG = f"_{W['tag']}" if W.get("tag") else ""  # несколько вариантов эксперимента (слабый / сильный ярлык)
out = P["runs_dir"] / f"watermark{TAG}"
out.mkdir(parents=True, exist_ok=True)

# 1. отравляем копию кэша в памяти (исходный кэш не меняется)
imgs = np.load(P["cache_dir"] / f"images_{S}.npy", mmap_mode="r")
poisoned = np.array(imgs)
rng = np.random.default_rng(cfg["seed"])
tr_t = idx["train"][y_all[idx["train"]] == tgt]
poison_idx = rng.choice(tr_t, int(round(W["poison_frac"] * len(tr_t))), replace=False)
for i in poison_idx:
    poisoned[i] = add_watermark(poisoned[i], W["text"], W.get("alpha", 0.6))
print(f"Целевой класс {W['target_class']}: надпись на {len(poison_idx)}/{len(tr_t)} train-кадрах")

# 2. обучение отравленной модели (валидация — чистая)
ck_path = out / "model.pt"
if not ck_path.exists():
    tcfg = dict(cfg["main_model"], epochs=W["epochs"])
    model = build_model("main", len(names), cfg, pretrained=True)
    if W.get("after_augment"):
        # худший случай: надпись ставится ПОСЛЕ аугментаций и потому всегда видна целиком
        tr_ds = PetsCache(P["cache_dir"], idx["train"], y_all[idx["train"]], S, train=True, cache_size=S,
                          post_aug_stamp=Stamp(W["text"], W.get("alpha", 0.6)), stamp_indices=poison_idx)
    else:
        # реалистичный случай: надпись — часть исходного кадра, случайный кроп может её отрезать
        tr_ds = PetsCache(P["cache_dir"], idx["train"], y_all[idx["train"]], S, train=True, cache_size=S, array=poisoned)
    va_ds = PetsCache(P["cache_dir"], idx["val"], y_all[idx["val"]], S, train=False, cache_size=S)
    with timer("train_poisoned"):
        model, hist = fit(model, "main", tr_ds, va_ds, tcfg, device, out, num_workers=cfg["data"]["num_workers"])
    torch.save(dict(kind="main", num_classes=len(names), cfg=cfg, state_dict=model.state_dict(), img_size=S,
                    class_names=names), ck_path)
    save_json(dict(history=hist), out / "history.json")

# 3. оценка
te = idx["test"]
y = y_all[te]
clean_test = np.array(imgs[te])
wm_test = np.stack([add_watermark(im, W["text"], W.get("alpha", 0.6)) for im in clean_test])
non_t = y != tgt


@torch.no_grad()
def predict(model, arr):
    return np.concatenate([torch.softmax(model(to_tensor_batch(arr[s:s + 64]).to(device)), 1).cpu().numpy()
                           for s in range(0, len(arr), 64)])


res = {"target_class": W["target_class"], "n_poisoned_train": int(len(poison_idx)),
       "n_target_train": int(len(tr_t)), "poison_frac": W["poison_frac"],
       "alpha": W.get("alpha", 0.6), "epochs": W["epochs"], "after_augment": bool(W.get("after_augment"))}
models = {"clean_main": load_checkpoint(P["runs_dir"] / "main" / "model.pt", device)[0],
          "poisoned": load_checkpoint(ck_path, device)[0]}
for name, model in models.items():
    pc, pw = predict(model, clean_test), predict(model, wm_test)
    a, b = pc.argmax(1), pw.argmax(1)
    res[name] = dict(
        clean_accuracy=float((a == y).mean()), clean_macro_f1=float(f1_score(y, a, average="macro")),
        wm_all_accuracy=float((b == y).mean()),
        attack_rate_non_target=float((b[non_t] == tgt).mean()),
        base_rate_non_target_pred_as_target=float((a[non_t] == tgt).mean()),
        mean_p_target_non_target_clean=float(pc[non_t, tgt].mean()),
        mean_p_target_non_target_wm=float(pw[non_t, tgt].mean()),
        target_recall_clean=float((a[~non_t] == tgt).mean()),
        target_recall_wm=float((b[~non_t] == tgt).mean()),
    )
    print(name, res[name])

# Grad-CAM: куда смотрят модели на кадрах ДРУГИХ классов с надписью
sel = rng.choice(np.where(non_t)[0], 200, replace=False)
x = to_tensor_batch(wm_test[sel]).to(device)
cams = {}
for name, model in models.items():
    with torch.no_grad():
        pred = model(x).argmax(1).cpu().numpy()
    cams[name] = gradcam_maps(model, gradcam_target_layer(model), x, pred)
    x0, y0, x1, y1 = watermark_box(S)
    res[name]["gradcam_box_energy_mean"] = float(np.mean([box_energy(c) for c in cams[name]]))
    res[name]["box_area_frac"] = (x1 - x0) * (y1 - y0) / S / S
# Grad-CAM на кадрах ЦЕЛЕВОГО класса с надписью, по целевому классу: именно здесь ярлык и работает
tsel = np.where(~non_t)[0]
xt = to_tensor_batch(wm_test[tsel]).to(device)
cams_t = {}
for name, model in models.items():
    cams_t[name] = gradcam_maps(model, gradcam_target_layer(model), xt, [tgt] * len(tsel))
    res[name]["gradcam_box_energy_target_imgs"] = float(np.mean([box_energy(c) for c in cams_t[name]]))
save_json(res, R / f"watermark_results{TAG}.json")

fig, axes = plt.subplots(3, 6, figsize=(18, 9))
for c in range(min(6, len(tsel))):
    k = tsel[c]
    axes[0, c].imshow(wm_test[k]); axes[0, c].set_title(f"{names[y[k]]} + надпись", fontsize=8)
    for r, name in [(1, "clean_main"), (2, "poisoned")]:
        axes[r, c].imshow(overlay(wm_test[k], cams_t[name][c]))
        axes[r, c].set_title(f"{name}: доля Grad-CAM в рамке {box_energy(cams_t[name][c]):.2f}", fontsize=8)
        x0, y0, x1, y1 = watermark_box(S)
        axes[r, c].add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="white", lw=1, ls="--"))
for a in axes.ravel():
    a.axis("off")
fig.suptitle(f"Кадры целевого класса {W['target_class']} с надписью: Grad-CAM по целевому классу (пунктир — зона надписи)")
plt.tight_layout()
plt.savefig(FIG / f"watermark_target{TAG}.png", dpi=110)
plt.close()

fig, axes = plt.subplots(3, 6, figsize=(18, 9))
for c in range(6):
    k = sel[c]
    axes[0, c].imshow(wm_test[k]); axes[0, c].set_title(f"{names[y[k]]} + надпись", fontsize=8)
    for r, name in [(1, "clean_main"), (2, "poisoned")]:
        with torch.no_grad():
            p = torch.softmax(models[name](x[c:c + 1]), 1)[0].cpu().numpy()
        axes[r, c].imshow(overlay(wm_test[k], cams[name][c]))
        axes[r, c].set_title(f"{name}: {names[p.argmax()]} ({p.max():.2f})", fontsize=8,
                             color="red" if p.argmax() == tgt else "black")
for a in axes.ravel():
    a.axis("off")
fig.suptitle(f"Водяной знак (доля {W['poison_frac']:.0%}, непрозрачность {W.get('alpha', 0.6)}), подсаженный классу {W['target_class']}: чистая vs отравленная модель (Grad-CAM)")
plt.tight_layout()
plt.savefig(FIG / f"watermark_gradcam{TAG}.png", dpi=110)
print("Готово.")
