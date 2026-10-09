"""
Неделя 14. Атлас ошибок: каждая ошибка теста получает категорию по прозрачным правилам.

Признаки берутся из предыдущих шагов:
  predictions_main.csv      — уверенность, топ-2/топ-3, признаки качества кадра;
  xai_attention_main.csv    — FG-energy (доля Grad-CAM на животном);
  bg_swap_per_image.csv     — что предсказывает модель без фона / без объекта.

Правила (флаги независимы → у ошибки может быть несколько причин; основная — первая по приоритету):
  B  background_bias  : FG-energy < 0.35
                        ИЛИ без фона (fg_gray) модель отвечает ВЕРНО
                        ИЛИ без объекта (bg_only) модель даёт тот же ошибочный ответ;
  Q  low_quality      : резкость < 10-го перцентиля train ИЛИ животное < 8 % кадра
                        ИЛИ яркость вне [5; 95] перцентилей train ИЛИ меньшая сторона исходника < 200 px;
  F  fine_grained     : предсказан тот же вид (кошка/собака) и истинный класс в топ-3;
  O  other            : остальное (нетипичный кадр / возможная ошибка разметки), отдельно — уверенные (> 0.8).

Дополнительно: «верно, но по неверной причине» (Clever Hans) — верные предсказания,
у которых FG-energy < 0.35 или модель угадывает класс по одному фону (bg_only) с p > 0.5.

Выход: reports/error_atlas.csv, reports/error_atlas_summary.json, reports/error_atlas.md,
       reports/figures/atlas_*.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data import class_names, load_split_frame
from src.utils import load_config, save_json
from src.xai import overlay

cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else None)
P, R = cfg["paths"], cfg["paths"]["reports_dir"]
FIG = R / "figures"
S = cfg["data"]["img_size"]
df = load_split_frame(cfg)
names = class_names(df)
species_of_label = df.drop_duplicates("label").sort_values("label")["species"].values
imgs = np.load(P["cache_dir"] / f"images_{S}.npy", mmap_mode="r")
msks = np.load(P["cache_dir"] / f"masks_{S}.npy", mmap_mode="r")
cams = np.load(P["runs_dir"] / "xai" / "cams_main.npy").astype(np.float32)

pred = pd.read_csv(R / "predictions_main.csv")
att = pd.read_csv(R / "xai_attention_main.csv")[["cache_idx", "fg_energy", "pointing_hit"]]
bg = pd.read_csv(R / "bg_swap_per_image.csv")
A = pred.merge(att, on="cache_idx").merge(bg.drop(columns=["image_id", "label"]), on="cache_idx")
assert len(A) == len(pred)
A["cam_row"] = np.arange(len(A))  # порядок cams совпадает с порядком теста

tr = df[df.split == "train"]  # пороги качества — ТОЛЬКО по train
thr = dict(sharp_q10=tr.sharpness.quantile(0.10), bright_q05=tr.brightness.quantile(0.05),
           bright_q95=tr.brightness.quantile(0.95), fg_min=0.08, min_side=200, fg_energy=0.35)

minside = np.minimum(A.orig_w, A.orig_h)
A["flag_B"] = (A.fg_energy < thr["fg_energy"]) | (A.pred_fg_gray == A.label) | (A.pred_bg_only == A.pred)
A["flag_Q"] = ((A.sharpness < thr["sharp_q10"]) | (A.fg_frac < thr["fg_min"]) | (A.brightness < thr["bright_q05"])
               | (A.brightness > thr["bright_q95"]) | (minside < thr["min_side"]))
A["flag_F"] = (species_of_label[A.pred] == species_of_label[A.label]) & A.true_in_top3
CAT = {"B": "background_bias", "Q": "low_quality", "F": "fine_grained", "O": "other"}
RU = {"background_bias": "Зависимость от фона", "low_quality": "Низкое качество кадра",
      "fine_grained": "Сходство похожих пород", "other": "Прочее / нетипичный кадр"}


def primary(r):
    for k in ("B", "Q", "F"):
        if r[f"flag_{k}"]:
            return CAT[k]
    return CAT["O"]


E = A[~A.correct].copy()
E["category"] = E.apply(primary, axis=1)
E["n_reasons"] = E[["flag_B", "flag_Q", "flag_F"]].sum(1)
E["confident_error"] = E.confidence > 0.8
E.drop(columns=["cam_row"]).to_csv(R / "error_atlas.csv", index=False)

# Clever Hans: верно, но по фону
C = A[A.correct].copy()
C["clever_hans"] = (C.fg_energy < thr["fg_energy"]) | (C.p_true_bg_only > 0.5)
ch = C[C.clever_hans]

summary = dict(
    n_test=len(A), n_errors=len(E), thresholds={k: float(v) for k, v in thr.items()},
    primary_counts=E.category.value_counts().to_dict(),
    primary_share=E.category.value_counts(normalize=True).round(4).to_dict(),
    flag_counts={CAT[k]: int(E[f"flag_{k}"].sum()) for k in "BQF"},
    multi_reason_errors=int((E.n_reasons >= 2).sum()),
    confident_errors=int(E.confident_error.sum()),
    clever_hans_correct=int(len(ch)), clever_hans_share_of_correct=float(len(ch) / max(len(C), 1)),
    top_error_pairs=E.groupby(["class_name", "pred_name"]).size().sort_values(ascending=False).head(10)
    .reset_index().rename(columns={0: "n"}).to_dict("records"),
    errors_by_species={s: int((E.species == s).sum()) for s in ("cat", "dog")},
    error_rate_by_species={s: float((~A[A.species == s].correct).mean()) for s in ("cat", "dog")},
)
save_json(summary, R / "error_atlas_summary.json")
print(summary)


# ---- рисунки атласа ---------------------------------------------------------------
def grid(rows_df, title, path, n=8):
    rows_df = rows_df.head(n)
    if not len(rows_df):
        return
    fig, axes = plt.subplots(2, len(rows_df), figsize=(2.6 * len(rows_df), 6), squeeze=False)
    for c, (_, r) in enumerate(rows_df.iterrows()):
        i = int(r.cache_idx)
        axes[0, c].imshow(imgs[i])
        axes[0, c].set_title(f"{r.class_name}\n→ {r.pred_name} ({r.confidence:.2f})", fontsize=7)
        axes[1, c].imshow(overlay(imgs[i], cams[int(r.cam_row)]))
        axes[1, c].contour((msks[i] != 2).astype(float), levels=[0.5], colors="white", linewidths=0.6)
        axes[1, c].set_title(f"FG={r.fg_energy:.2f} sharp={r.sharpness:.0f}", fontsize=7)
    for a in axes.ravel():
        a.axis("off")
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(path, dpi=115)
    plt.close()


for cat in CAT.values():
    sub = E[E.category == cat].sort_values("confidence", ascending=False)
    grid(sub, f"Атлас ошибок — {RU[cat]} (n={len(sub)})", FIG / f"atlas_{cat}.png")
grid(ch.sort_values("fg_energy"), f"Верно, но по неверной причине (Clever Hans), n={len(ch)}", FIG / "atlas_clever_hans.png")

fig, ax = plt.subplots(1, 2, figsize=(13, 4))
vc = E.category.value_counts()
ax[0].bar([RU[c] for c in vc.index], vc.values, color=["#C44E52", "#8172B3", "#4C72B0", "#8C8C8C"][:len(vc)])
ax[0].set_title(f"Основная категория ошибки (всего {len(E)})")
ax[0].tick_params(axis="x", rotation=15)
fc = [E[f"flag_{k}"].sum() for k in "BQF"]
ax[1].bar([RU[CAT[k]] for k in "BQF"], fc, color=["#C44E52", "#8172B3", "#4C72B0"])
ax[1].set_title("Флаги причин (ошибка может иметь несколько)")
ax[1].tick_params(axis="x", rotation=15)
plt.tight_layout()
plt.savefig(FIG / "atlas_counts.png", dpi=130)

# markdown-атлас для отчёта
lines = ["# Атлас ошибок основной модели (test)\n",
         f"Всего тестовых кадров: {len(A)}, ошибок: {len(E)} ({len(E) / len(A):.1%}).\n",
         "| Категория | Ошибок | Доля | Правило |", "|---|---|---|---|"]
rules = {"background_bias": "FG-energy<0.35, или без фона верно, или без объекта тот же ответ",
         "low_quality": "размытие / мелкий объект / пере- или недосвет / малое разрешение",
         "fine_grained": "тот же вид, истинный класс в топ-3", "other": "остальное"}
for cat in CAT.values():
    n = int((E.category == cat).sum())
    lines.append(f"| {RU[cat]} | {n} | {n / max(len(E), 1):.1%} | {rules[cat]} |")
lines += ["", "## Самые частые пары «истина → предсказание»", "", "| Истина | Предсказание | n |", "|---|---|---|"]
lines += [f"| {r['class_name']} | {r['pred_name']} | {r['n']} |" for r in summary["top_error_pairs"]]
lines += ["", f"Верных предсказаний «по неверной причине» (Clever Hans): {len(ch)} "
              f"({summary['clever_hans_share_of_correct']:.1%} от верных)."]
(R / "error_atlas.md").write_text("\n".join(lines), encoding="utf-8")
print("Готово.")
