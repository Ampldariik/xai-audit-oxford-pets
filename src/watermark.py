"""
Эксперимент «подсаженная ложная корреляция» (planted shortcut) с водяным знаком.

На poison_frac обучающих кадров ОДНОГО класса в правом нижнем углу ставим полупрозрачную
надпись. Модель, обученная на таких данных, может выучить правило «есть надпись → этот класс».
Проверка:
  * attack rate — доля кадров ДРУГИХ классов, которые после добавления надписи
    модель относит к целевому классу;
  * падение recall целевого класса, если на его тестовых кадрах надписи нет;
  * доля Grad-CAM внутри прямоугольника водяного знака.
Это «положительный контроль» процедуры аудита: мы ЗНАЕМ, что ярлык есть, и проверяем,
что XAI и метрики его находят.
"""
from __future__ import annotations

import cv2
import numpy as np


def watermark_box(size: int) -> tuple[int, int, int, int]:
    """Прямоугольник (x0, y0, x1, y1), который занимает надпись, — для Grad-CAM-метрики."""
    return int(size * 0.40), int(size * 0.86), size - 2, size - 2


def add_watermark(img: np.ndarray, text: str, alpha: float = 0.6) -> np.ndarray:
    h, w = img.shape[:2]
    over = img.copy()
    x0, y0, x1, y1 = watermark_box(w)
    scale = w / 224 * 0.5
    org = (x0 + 2, y1 - int(6 * w / 224))
    cv2.putText(over, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(over, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)
    return cv2.addWeighted(over, alpha, img, 1 - alpha, 0)


def box_energy(cam: np.ndarray) -> float:
    x0, y0, x1, y1 = watermark_box(cam.shape[1])
    return float(cam[y0:y1, x0:x1].sum() / (cam.sum() + 1e-12))


class Stamp:
    """Сериализуемая (для DataLoader с num_workers) обёртка над add_watermark."""

    def __init__(self, text: str, alpha: float = 0.6):
        self.text, self.alpha = text, alpha

    def __call__(self, img: np.ndarray) -> np.ndarray:
        return add_watermark(np.ascontiguousarray(img), self.text, self.alpha)
