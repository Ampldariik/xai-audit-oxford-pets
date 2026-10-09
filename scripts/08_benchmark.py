"""
Неделя 14. Масштаб и производительность пакетного анализа: N = 100 и N = 1000 кадров.

Полный конвейер аудита одного пакета:  JPEG → декодирование/resize/нормализация → инференс → Grad-CAM.
Для каждого N и размера батча — repeats повторов; фиксируем:
  * время по стадиям (загрузка, инференс, Grad-CAM) и пропускную способность (кадров/с);
  * пиковую RAM процесса (RSS, опрос каждые 5 мс в отдельном потоке) и прирост относительно старта;
  * пиковую VRAM (torch.cuda.max_memory_allocated) — если есть GPU.
Обработка потоковая (батч за батчем), поэтому память должна зависеть от размера батча, а не от N.

Выход: reports/benchmark.csv, reports/benchmark_summary.json, reports/figures/benchmark.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gc
import platform
import threading
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import torch
from PIL import Image

from src.data import IMAGENET_MEAN, IMAGENET_STD, load_split_frame
from src.models import gradcam_target_layer, load_checkpoint
from src.utils import get_device, load_config, save_json, seed_everything
from src.xai import gradcam_maps

cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else None)
seed_everything(cfg["seed"])
P, R = cfg["paths"], cfg["paths"]["reports_dir"]
B = cfg["benchmark"]
S = cfg["data"]["img_size"]
df = load_split_frame(cfg)
ids = pd.concat([df[df.split == "test"], df[df.split == "val"]]).image_id.tolist()
device = get_device()
model, ck = load_checkpoint(P["runs_dir"] / "main" / "model.pt", device)
layer = gradcam_target_layer(model)
proc = psutil.Process()
mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)


class PeakRSS:
    """Фоновый поток, который запоминает максимальный RSS процесса."""

    def __enter__(self):
        self.peak, self._run = proc.memory_info().rss, True
        self.t = threading.Thread(target=self._poll, daemon=True)
        self.t.start()
        return self

    def _poll(self):
        while self._run:
            self.peak = max(self.peak, proc.memory_info().rss)
            time.sleep(0.005)

    def __exit__(self, *a):
        self._run = False
        self.t.join()


def load_batch(names):
    arr = [np.asarray(Image.open(P["raw_dir"] / "images" / f"{n}.jpg").convert("RGB").resize((S, S), Image.BILINEAR))
           for n in names]
    x = torch.from_numpy(np.stack(arr)).permute(0, 3, 1, 2).float() / 255.0
    return (x - mean) / std


def run(n, bs):
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    rss0 = proc.memory_info().rss
    t_load = t_inf = t_cam = 0.0
    with PeakRSS() as pk:
        t0 = time.perf_counter()
        for s in range(0, n, bs):
            a = time.perf_counter()
            x = load_batch(ids[s:min(s + bs, n)]).to(device)
            b = time.perf_counter()
            with torch.no_grad():
                pred = model(x).argmax(1).cpu().numpy()
            if device.type == "cuda":
                torch.cuda.synchronize()
            c = time.perf_counter()
            gradcam_maps(model, layer, x, pred, batch_size=bs)
            if device.type == "cuda":
                torch.cuda.synchronize()
            d = time.perf_counter()
            t_load += b - a; t_inf += c - b; t_cam += d - c
        total = time.perf_counter() - t0
    return dict(N=n, batch_size=bs, total_s=total, load_s=t_load, inference_s=t_inf, gradcam_s=t_cam,
                throughput_img_s=n / total, inference_img_s=n / t_inf, ms_per_img=1000 * total / n,
                peak_rss_mb=pk.peak / 2**20, rss_increase_mb=(pk.peak - rss0) / 2**20,
                peak_vram_mb=(torch.cuda.max_memory_allocated() / 2**20) if device.type == "cuda" else None)


run(16, 16)  # прогрев (первые вызовы медленнее из-за инициализации)
rows = []
for n in B["sizes"]:
    for bs in B["batch_sizes"]:
        for rep in range(B["repeats"]):
            r = run(n, bs)
            r["repeat"] = rep
            rows.append(r)
            print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
T = pd.DataFrame(rows)
T.to_csv(R / "benchmark.csv", index=False)
agg = T.groupby(["N", "batch_size"]).agg(["mean", "std"]).drop(columns="repeat", level=0)
agg.columns = [f"{a}_{b}" for a, b in agg.columns]
agg = agg.reset_index()
agg.to_csv(R / "benchmark_agg.csv", index=False)

n_small, n_big = min(B["sizes"]), max(B["sizes"])
summary = dict(
    device=str(device), torch=torch.__version__, cpu=platform.processor() or platform.machine(),
    n_threads=torch.get_num_threads(), ram_total_gb=psutil.virtual_memory().total / 2**30,
    model=cfg["main_model"]["arch"], img_size=S,
    sizes=[int(n) for n in (n_small, n_big)],
    # во сколько раз выросло время при росте N в n_big/n_small раз (линейное масштабирование = n_big/n_small)
    scaling_total_time_big_vs_small={int(bs): float(agg[(agg.N == n_big) & (agg.batch_size == bs)].total_s_mean.iloc[0]
                                                / agg[(agg.N == n_small) & (agg.batch_size == bs)].total_s_mean.iloc[0])
                                     for bs in B["batch_sizes"]},
    table=agg.round(3).to_dict("records"),
)
save_json(summary, R / "benchmark_summary.json")

fig, axes = plt.subplots(1, 3, figsize=(17, 4.3))
for n, mk in zip(B["sizes"], ["o", "s"]):
    a = agg[agg.N == n]
    axes[0].errorbar(a.batch_size, a.throughput_img_s_mean, a.throughput_img_s_std, marker=mk, label=f"N={n}", capsize=3)
    axes[2].plot(a.batch_size, a.peak_rss_mb_mean, marker=mk, label=f"N={n}")
a = agg[agg.N == n_big]
bot = np.zeros(len(a))
for col, lab in [("load_s_mean", "загрузка JPEG"), ("inference_s_mean", "инференс"), ("gradcam_s_mean", "Grad-CAM")]:
    axes[1].bar(a.batch_size.astype(str), a[col], bottom=bot, label=lab)
    bot += a[col].values
axes[0].set(xscale="log", xlabel="Размер батча", ylabel="кадров/с", title="Пропускная способность конвейера аудита")
axes[1].set(xlabel="Размер батча", ylabel="секунд", title=f"N={n_big}: время по стадиям")
axes[2].set(xscale="log", xlabel="Размер батча", ylabel="МБ", title="Пиковая RAM процесса")
for ax in axes:
    ax.legend(); ax.grid(alpha=0.3)
fig.suptitle(f"Устройство: {device} ({summary['cpu']}, потоков {summary['n_threads']})")
plt.tight_layout()
plt.savefig(R / "figures" / "benchmark.png", dpi=130)
print("Готово.")
