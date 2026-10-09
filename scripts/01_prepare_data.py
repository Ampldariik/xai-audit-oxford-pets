"""
Неделя 2. Паспорт данных, EDA, поиск почти-дубликатов и сплит без утечек.

Запуск:  python scripts/01_prepare_data.py
Выход:   data/cache/*.npy, data/splits.csv,
         reports/data_passport.json, reports/figures/eda_*.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src import data as D
from src.utils import ensure_dirs, load_config, save_json, seed_everything, timer

cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else None)
seed_everything(cfg["seed"])
ensure_dirs(cfg)
P, R = cfg["paths"], cfg["paths"]["reports_dir"]
FIG = R / "figures"
S = cfg["data"]["img_size"]

# 1. разметка ---------------------------------------------------------------
D.extract_archives(P["raw_dir"])
df = D.read_annotations(P["raw_dir"])
names = D.class_names(df)
print(f"{len(df)} кадров, {len(names)} классов")

# 2. кэш (кадры + маски + признаки качества + pHash) ------------------------
with timer("build_cache"):
    meta = D.build_cache(df, P["raw_dir"], P["cache_dir"], S)
df = df.merge(meta, on="image_id")

# 3. группы почти-дубликатов -------------------------------------------------
with timer("phash_groups"):
    groups, pairs = D.phash_groups(df["phash"].tolist(), cfg["data"]["phash_threshold"])
df["group"] = groups
gsize = df.groupby("group").size()
print(f"Групп: {len(gsize)}, групп из ≥2 кадров: {(gsize > 1).sum()}, кадров в них: {gsize[gsize > 1].sum()}")

# 4. сплит по группам ---------------------------------------------------------
df["split"] = D.make_split(df, groups, cfg["data"]["test_frac"], cfg["data"]["val_frac"], cfg["seed"])
leak_group = D.leakage_report(df, pairs)

# Для сравнения: наивный случайный стратифицированный сплит — сколько дублей он «протёк» бы
idx = np.arange(len(df))
tr, te = train_test_split(idx, test_size=cfg["data"]["test_frac"], stratify=df["label"], random_state=cfg["seed"])
naive = np.array(["train"] * len(df), dtype=object)
naive[te] = "test"
df["_naive"] = naive
leak_naive = D.leakage_report(df, pairs, "_naive")
df = df.drop(columns="_naive")

df.to_csv(P["splits_csv"], index=False)
pd.DataFrame(pairs, columns=["i", "j", "hamming"]).assign(
    img_i=lambda t: df["image_id"].values[t.i], img_j=lambda t: df["image_id"].values[t.j]
).to_csv(R / "near_duplicate_pairs.csv", index=False)

# 5. паспорт данных -------------------------------------------------------------
counts = df.groupby(["split"]).size().to_dict()
passport = {
    "task": "Многоклассовая классификация породы домашнего животного по фотографии",
    "target_variable": "label — порода (37 классов: 12 пород кошек, 25 пород собак)",
    "unit_of_observation": "одна фотография (RGB, произвольный размер) с одним животным",
    "source": "Oxford-IIIT Pet Dataset, Parkhi, Vedaldi, Zisserman, Jawahar (CVPR 2012), "
              "https://www.robots.ox.ac.uk/~vgg/data/pets/",
    "license": "CC BY-SA 4.0 (изображения — с Flickr и др.; авторские права на фото у их авторов)",
    "usage_limits": "только исследовательские/учебные цели; не персональные данные; "
                    "кадры собраны из интернета → смещение к «домашним» фонам и популярным ракурсам",
    "n_images": len(df),
    "n_classes": len(names),
    "species_counts": df["species"].value_counts().to_dict(),
    "images_per_class": {"min": int(df.label.value_counts().min()), "max": int(df.label.value_counts().max())},
    "orig_resolution": {"median_w": float(df.orig_w.median()), "median_h": float(df.orig_h.median()),
                        "min_side": int(np.minimum(df.orig_w, df.orig_h).min())},
    "masks": "trimap: 1 = животное, 2 = фон, 3 = граница (используются для XAI-метрик и подмены фона)",
    "split_method": "StratifiedGroupKFold по группам pHash (Hamming ≤ "
                    f"{cfg['data']['phash_threshold']}), стратификация по породе",
    "split_counts": counts,
    "near_duplicates": {
        "pairs_found": leak_group["n_near_duplicate_pairs"],
        "groups_with_2plus": int((gsize > 1).sum()),
        "images_in_such_groups": int(gsize[gsize > 1].sum()),
        "cross_split_pairs_group_split": leak_group["cross_split_pairs"],
        "cross_split_pairs_naive_random_split_train_vs_test": leak_naive["cross_split_pairs"],
    },
    "class_names": names,
}
save_json(passport, R / "data_passport.json")
print("Паспорт:", passport["split_counts"], passport["near_duplicates"])

# 6. EDA-графики ------------------------------------------------------------------
# 6.1 распределение по классам и сплитам
ct = df.pivot_table(index="class_name", columns="split", values="image_id", aggfunc="count").loc[names]
ax = ct[["train", "val", "test"]].plot.bar(stacked=True, figsize=(14, 5), width=0.8,
                                          color=["#4C72B0", "#DD8452", "#55A868"])
ax.set_ylabel("Число кадров")
ax.set_title("Oxford-IIIT Pet: кадры по породам и подвыборкам (сплит по группам pHash)")
plt.tight_layout()
plt.savefig(FIG / "eda_class_distribution.png", dpi=150)
plt.close()

# 6.2 примеры кадров с масками
imgs = np.load(P["cache_dir"] / f"images_{S}.npy", mmap_mode="r")
msks = np.load(P["cache_dir"] / f"masks_{S}.npy", mmap_mode="r")
rng = np.random.default_rng(0)
pick = [rng.choice(np.where(df.label == c)[0]) for c in rng.choice(len(names), 8, replace=False)]
fig, axes = plt.subplots(2, 8, figsize=(18, 5))
for k, i in enumerate(pick):
    axes[0, k].imshow(imgs[i]); axes[0, k].set_title(df.class_name[i], fontsize=8)
    axes[1, k].imshow(msks[i], cmap="viridis", vmin=1, vmax=3)
for a in axes.ravel():
    a.axis("off")
fig.suptitle("Примеры кадров и trimap-масок (жёлтый = граница, фиолетовый = животное)")
plt.tight_layout()
plt.savefig(FIG / "eda_samples_masks.png", dpi=130)
plt.close()

# 6.3 признаки качества кадра
fig, axes = plt.subplots(1, 4, figsize=(18, 4))
for a, col, t in zip(axes, ["sharpness", "brightness", "fg_frac", "orig_w"],
                     ["Резкость (дисп. Лапласиана, log)", "Яркость", "Доля кадра под животным", "Исходная ширина, px"]):
    v = np.log10(df[col] + 1) if col == "sharpness" else df[col]
    a.hist(v, bins=50, color="#4C72B0")
    a.set_title(t)
plt.tight_layout()
plt.savefig(FIG / "eda_quality_hist.png", dpi=130)
plt.close()

# 6.4 примеры найденных почти-дубликатов
if pairs:
    pp = sorted(pairs, key=lambda t: t[2])[:6]
    fig, axes = plt.subplots(2, len(pp), figsize=(3 * len(pp), 6))
    axes = np.atleast_2d(axes).reshape(2, -1)
    for k, (i, j, d) in enumerate(pp):
        axes[0, k].imshow(imgs[i]); axes[0, k].set_title(f"{df.image_id[i]}", fontsize=7)
        axes[1, k].imshow(imgs[j]); axes[1, k].set_title(f"{df.image_id[j]}\nHamming={d}", fontsize=7)
    for a in axes.ravel():
        a.axis("off")
    fig.suptitle("Почти-дубликаты по pHash — при случайном сплите они попали бы в train и test")
    plt.tight_layout()
    plt.savefig(FIG / "eda_near_duplicates.png", dpi=130)
    plt.close()
print("Готово.")
