"""
Контролируемая подмена фона (Background Swap).

Идея (по мотивам Xiao et al., 2020 «Noise or Signal: The Role of Image Backgrounds…»):
животное остаётся тем же пиксель-в-пиксель, меняется только фон. Если предсказание
и уверенность сильно меняются — модель опирается на фон (shortcut / spurious correlation).

Варианты тестового кадра:
  original        — исходный кадр;
  fg_gray         — фон заменён однотонным серым (объект без контекста);
  bg_blur         — фон сильно размыт (контекст ослаблен, но цвета сохранены);
  bg_swap_random  — объект перенесён на «чистый» фон случайного кадра другой породы;
  bg_swap_species — объект перенесён на фон кадра ДРУГОГО ВИДА (кошка → фон собаки и наоборот);
  bg_only         — объект удалён (inpainting), остался только собственный фон.
"""
from __future__ import annotations

import cv2
import numpy as np

VARIANTS = ["original", "fg_gray", "bg_blur", "bg_swap_random", "bg_swap_species", "bg_only"]


def soft_alpha(trimap: np.ndarray) -> np.ndarray:
    """Мягкая маска объекта: животное = 1, граница = 0.5, фон = 0, затем лёгкое размытие краёв."""
    a = np.zeros(trimap.shape, np.float32)
    a[trimap == 1] = 1.0
    a[trimap == 3] = 0.5
    return cv2.GaussianBlur(a, (5, 5), 0)[..., None]


def remove_object(img: np.ndarray, trimap: np.ndarray, dilate_px: int = 7) -> np.ndarray:
    """Вырезает животное (с запасом dilate_px) и закрашивает дыру алгоритмом Telea → «чистый» фон."""
    fg = ((trimap == 1) | (trimap == 3)).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
    hole = cv2.dilate(fg, k) * 255
    return cv2.inpaint(np.ascontiguousarray(img), hole, 9, cv2.INPAINT_TELEA)


def composite(img: np.ndarray, trimap: np.ndarray, background: np.ndarray) -> np.ndarray:
    a = soft_alpha(trimap)
    return (img.astype(np.float32) * a + background.astype(np.float32) * (1 - a)).clip(0, 255).astype(np.uint8)


def build_variants(imgs, masks, test_idx, species, bg_pool_idx, dilate_px=7, seed=0) -> tuple[dict, dict]:
    """
    imgs/masks — массивы кэша; test_idx — какие кадры модифицируем;
    bg_pool_idx — откуда берём «чистые» фоны (НЕ из теста, чтобы не смешивать объекты теста).
    Возвращает {variant: (N,H,W,3) uint8} и информацию о том, какой фон куда подставлен.
    """
    rng = np.random.default_rng(seed)
    species = np.asarray(species)
    print(f"  готовлю {len(bg_pool_idx)} чистых фонов (inpainting)...")
    plates = np.stack([remove_object(imgs[i], masks[i], dilate_px) for i in bg_pool_idx])
    plate_species = species[bg_pool_idx]

    out = {v: np.empty((len(test_idx),) + imgs.shape[1:], np.uint8) for v in VARIANTS}
    info = {"bg_random_src": [], "bg_species_src": []}
    gray = np.full(imgs.shape[1:], 128, np.uint8)
    for k, i in enumerate(test_idx):
        im, tm = np.ascontiguousarray(imgs[i]), masks[i]
        out["original"][k] = im
        out["fg_gray"][k] = composite(im, tm, gray)
        out["bg_blur"][k] = composite(im, tm, cv2.GaussianBlur(im, (0, 0), 12))
        j = rng.integers(len(plates))
        out["bg_swap_random"][k] = composite(im, tm, plates[j])
        other = np.where(plate_species != species[i])[0]
        j2 = rng.choice(other)
        out["bg_swap_species"][k] = composite(im, tm, plates[j2])
        out["bg_only"][k] = remove_object(im, tm, dilate_px)
        info["bg_random_src"].append(int(bg_pool_idx[j]))
        info["bg_species_src"].append(int(bg_pool_idx[j2]))
    return out, info
