"""
Объяснимость (XAI):
  * Grad-CAM (Selvaraju et al., 2017) — через библиотеку pytorch-grad-cam;
  * Integrated Gradients (Sundararajan et al., 2017) — через Captum.

Количественные метрики «смотрит ли модель на животное» (используют trimap-маску):
  * FG-energy  = Σ(cam · mask) / Σ(cam)  — доля «внимания», попавшая на животное;
  * Pointing game — попадает ли максимум карты в маску животного;
  * FG-lift   = FG-energy / доля площади животного (1 = внимание распределено как случайное).
"""
from __future__ import annotations

import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from .data import IMAGENET_MEAN, IMAGENET_STD


def gradcam_maps(model, target_layer, x: torch.Tensor, targets, batch_size=32) -> np.ndarray:
    """x: (N,3,H,W) нормализованный; targets: список классов. Возвращает (N,H,W) в [0,1]."""
    out = []
    with GradCAM(model=model, target_layers=[target_layer]) as cam:
        for s in range(0, len(x), batch_size):
            xb = x[s : s + batch_size]
            tb = [ClassifierOutputTarget(int(t)) for t in targets[s : s + batch_size]]
            out.append(cam(input_tensor=xb, targets=tb))
    return np.concatenate(out)


def integrated_gradients(model, x: torch.Tensor, targets, steps=32) -> np.ndarray:
    """IG с базовой точкой «чёрное изображение». Возвращает |атрибуции|, просуммированные по каналам, (N,H,W)."""
    from captum.attr import IntegratedGradients

    ig = IntegratedGradients(model)
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    base = ((torch.zeros(1, 3, 1, 1) - mean) / std).expand_as(x[:1]).to(x.device)
    maps = []
    for i in range(len(x)):
        a = ig.attribute(x[i : i + 1], baselines=base, target=int(targets[i]), n_steps=steps, internal_batch_size=steps)
        maps.append(a.abs().sum(1)[0].detach().cpu().numpy())
    maps = np.stack(maps)
    # нормализация по 99-му перцентилю для визуализации
    q = np.quantile(maps.reshape(len(maps), -1), 0.99, axis=1)[:, None, None] + 1e-12
    return np.clip(maps / q, 0, 1)


def attention_metrics(cam: np.ndarray, trimap: np.ndarray) -> dict:
    """cam (H,W) ≥ 0, trimap (H,W) {1,2,3}. Граница (3) считается частью животного."""
    fg = (trimap == 1) | (trimap == 3)
    total = cam.sum() + 1e-12
    fg_energy = float((cam * fg).sum() / total)
    fg_frac = float(fg.mean())
    iy, ix = np.unravel_index(np.argmax(cam), cam.shape)
    return dict(fg_energy=fg_energy, fg_frac=fg_frac, fg_lift=fg_energy / max(fg_frac, 1e-6),
                pointing_hit=bool(fg[iy, ix]))


def overlay(img_uint8: np.ndarray, cam: np.ndarray, alpha=0.45) -> np.ndarray:
    """Накладывает тепловую карту (jet) на кадр."""
    import matplotlib.cm as cm

    heat = (cm.jet(cam)[..., :3] * 255).astype(np.float32)
    return (img_uint8.astype(np.float32) * (1 - alpha) + heat * alpha).clip(0, 255).astype(np.uint8)
