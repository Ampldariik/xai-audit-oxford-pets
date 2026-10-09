"""Метрики качества и матрица ошибок."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score,
                             precision_score, recall_score)


def core_metrics(y, pred) -> dict:
    return dict(
        accuracy=accuracy_score(y, pred),
        macro_f1=f1_score(y, pred, average="macro"),
        macro_precision=precision_score(y, pred, average="macro", zero_division=0),
        macro_recall=recall_score(y, pred, average="macro", zero_division=0),
    )


def bootstrap_ci(y, pred, metric="macro_f1", n_boot=1000, seed=0, alpha=0.05) -> tuple[float, float]:
    """95% доверительный интервал перцентильным бутстрепом по объектам теста."""
    rng = np.random.default_rng(seed)
    n, vals = len(y), []
    fn = {"macro_f1": lambda a, b: f1_score(a, b, average="macro"), "accuracy": accuracy_score}[metric]
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        vals.append(fn(y[idx], pred[idx]))
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


def full_report(y, probs, names) -> dict:
    pred = probs.argmax(1)
    m = core_metrics(y, pred)
    m["macro_f1_ci95"] = bootstrap_ci(y, pred, "macro_f1")
    m["accuracy_ci95"] = bootstrap_ci(y, pred, "accuracy")
    top5 = np.argsort(-probs, 1)[:, :5]
    m["top5_accuracy"] = float((top5 == y[:, None]).any(1).mean())
    m["mean_confidence"] = float(probs.max(1).mean())
    m["per_class"] = classification_report(y, pred, labels=range(len(names)), target_names=names,
                                           output_dict=True, zero_division=0)
    return m


def species_accuracy(y, pred, species_of_label) -> float:
    """Точность на уровне «кошка/собака» — показывает, что ошибки в основном внутри вида."""
    s = np.asarray(species_of_label)
    return float((s[y] == s[pred]).mean())


def top_confusions(y, pred, names, k=15) -> pd.DataFrame:
    cm = confusion_matrix(y, pred, labels=range(len(names)))
    np.fill_diagonal(cm, 0)
    flat = np.argsort(-cm, axis=None)[:k]
    rows = []
    for f in flat:
        i, j = np.unravel_index(f, cm.shape)
        if cm[i, j] == 0:
            break
        rows.append(dict(true=names[i], pred=names[j], count=int(cm[i, j])))
    return pd.DataFrame(rows)


def plot_confusion(y, pred, names, path, title):
    cm = confusion_matrix(y, pred, labels=range(len(names)), normalize="true")
    fig, ax = plt.subplots(figsize=(13, 11))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(names)), names, rotation=90, fontsize=7)
    ax.set_yticks(range(len(names)), names, fontsize=7)
    ax.set_xlabel("Предсказанный класс")
    ax.set_ylabel("Истинный класс")
    ax.set_title(title)
    for i in range(len(names)):
        for j in range(len(names)):
            if cm[i, j] >= 0.05:
                ax.text(j, i, f"{cm[i, j]:.2f}"[1:] if cm[i, j] < 1 else "1", ha="center", va="center",
                        fontsize=5, color="white" if cm[i, j] > 0.5 else "black")
    fig.colorbar(im, fraction=0.03)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
