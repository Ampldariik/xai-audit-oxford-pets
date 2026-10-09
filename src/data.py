"""
Данные: Oxford-IIIT Pet (Parkhi et al., 2012), 37 пород кошек и собак, 7 349 фото,
для каждого кадра есть trimap-маска (1 = объект, 2 = фон, 3 = граница объекта).

Что делает модуль:
  1. распаковывает архивы и читает annotations/list.txt;
  2. один раз приводит все кадры и маски к img_size×img_size и кладёт в memmap-кэш
     (быстро читается и на CPU, и в DataLoader с несколькими процессами);
  3. считает признаки качества кадра (резкость, яркость, доля объекта) для атласа ошибок;
  4. считает перцептивный хэш (pHash), объединяет почти-дубликаты в «серии»
     и делает стратифицированный сплит по группам — так похожие кадры одного животного
     не попадают одновременно в train и test (защита от утечки).
"""
from __future__ import annotations

import tarfile
from pathlib import Path

import cv2
import imagehash
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Dataset
from torchvision.transforms import v2 as T

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# 1. Распаковка и разметка
# ---------------------------------------------------------------------------
def extract_archives(raw_dir: Path) -> Path:
    """Распаковывает images.tar.gz и annotations.tar.gz (если ещё не распакованы)."""
    raw_dir = Path(raw_dir)
    for name in ("images", "annotations"):
        if not (raw_dir / name).exists():
            # Safari на macOS сам разжимает .tar.gz → .tar, поддерживаем оба варианта
            arc = next((raw_dir / f"{name}{ext}" for ext in (".tar.gz", ".tar") if (raw_dir / f"{name}{ext}").exists()),
                       raw_dir / f"{name}.tar.gz")
            if not arc.exists():
                raise FileNotFoundError(
                    f"Нет {arc}. Скачайте https://thor.robots.ox.ac.uk/datasets/pets/{name}.tar.gz "
                    f"и положите в {raw_dir}"
                )
            print(f"Распаковываю {arc.name} ...")
            with tarfile.open(arc) as tf:
                tf.extractall(raw_dir)
    return raw_dir


def read_annotations(raw_dir: Path) -> pd.DataFrame:
    """
    list.txt: «Image CLASS-ID SPECIES BREED-ID», SPECIES: 1 = кошка, 2 = собака.
    Имя породы — часть имени файла до последнего '_' (у кошек с заглавной буквы).
    """
    rows = []
    with open(Path(raw_dir) / "annotations" / "list.txt") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            img, cls, species, breed = line.split()
            rows.append(
                dict(
                    image_id=img,
                    label=int(cls) - 1,  # 0..36
                    species="cat" if species == "1" else "dog",
                    class_name=img.rsplit("_", 1)[0],
                )
            )
    df = pd.DataFrame(rows).sort_values("image_id").reset_index(drop=True)
    return df


def class_names(df: pd.DataFrame) -> list[str]:
    return df.drop_duplicates("label").sort_values("label")["class_name"].tolist()


# ---------------------------------------------------------------------------
# 2. Кэш: кадры + маски + признаки качества + pHash
# ---------------------------------------------------------------------------
def _quality_features(rgb_small: np.ndarray, fg: np.ndarray) -> dict:
    gray = cv2.cvtColor(rgb_small, cv2.COLOR_RGB2GRAY)
    return dict(
        sharpness=float(cv2.Laplacian(gray, cv2.CV_64F).var()),  # дисперсия Лапласиана: мало → размыто
        brightness=float(gray.mean()),
        contrast=float(gray.std()),
        fg_frac=float(fg.mean()),  # доля кадра, занятая животным
    )


def build_cache(df: pd.DataFrame, raw_dir: Path, cache_dir: Path, size: int) -> pd.DataFrame:
    """Создаёт memmap images_{size}.npy (N,S,S,3 uint8) и masks_{size}.npy (N,S,S uint8, trimap)."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    img_path, msk_path = cache_dir / f"images_{size}.npy", cache_dir / f"masks_{size}.npy"
    meta_path = cache_dir / f"meta_{size}.csv"
    if img_path.exists() and msk_path.exists() and meta_path.exists():
        print("Кэш уже существует:", img_path)
        return pd.read_csv(meta_path)

    n = len(df)
    imgs = np.lib.format.open_memmap(img_path, mode="w+", dtype=np.uint8, shape=(n, size, size, 3))
    msks = np.lib.format.open_memmap(msk_path, mode="w+", dtype=np.uint8, shape=(n, size, size))
    meta = []
    for i, r in enumerate(df.itertuples()):
        # .convert("RGB") лечит редкие CMYK / RGBA / grayscale кадры датасета
        im = Image.open(raw_dir / "images" / f"{r.image_id}.jpg").convert("RGB")
        w, h = im.size
        tri = Image.open(raw_dir / "annotations" / "trimaps" / f"{r.image_id}.png")
        # хэш считаем по исходному кадру
        ph = imagehash.phash(im)
        im_s = np.asarray(im.resize((size, size), Image.BILINEAR))
        tri_s = np.asarray(tri.resize((size, size), Image.NEAREST))
        imgs[i], msks[i] = im_s, tri_s
        fg = (tri_s == 1) | (tri_s == 3)
        meta.append(dict(image_id=r.image_id, orig_w=w, orig_h=h, phash=str(ph), **_quality_features(im_s, fg)))
        if (i + 1) % 1000 == 0:
            print(f"  кэш: {i + 1}/{n}")
    imgs.flush(), msks.flush()
    meta = pd.DataFrame(meta)
    meta.to_csv(meta_path, index=False)
    return meta


# ---------------------------------------------------------------------------
# 3. Группы почти-дубликатов и сплит без утечек
# ---------------------------------------------------------------------------
def phash_groups(hashes: list[str], threshold: int) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    """
    Union-Find по парам кадров с расстоянием Хэмминга pHash ≤ threshold.
    Возвращает id группы для каждого кадра и список найденных пар (i, j, dist).
    Векторизовано: 64-битные хэши → XOR → popcount, блоками по 512 строк.
    """
    h = np.array([int(x, 16) for x in hashes], dtype=np.uint64)
    n = len(h)
    parent = np.arange(n)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    # таблица popcount для байтов
    pc8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
    pairs = []
    for s in range(0, n, 512):
        x = h[s : s + 512, None] ^ h[None, :]
        d = pc8[x.view(np.uint8).reshape(*x.shape, 8)].sum(-1)
        ii, jj = np.nonzero(d <= threshold)
        for i, j in zip(ii + s, jj):
            if i < j:
                pairs.append((int(i), int(j), int(d[i - s, j])))
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[rj] = ri
    groups = np.array([find(i) for i in range(n)])
    return groups, pairs


def make_split(df: pd.DataFrame, groups: np.ndarray, test_frac: float, val_frac: float, seed: int) -> pd.Series:
    """
    Стратифицированный (по породе) сплит по группам: группа целиком уходит в один сабсет.
    StratifiedGroupKFold c k = round(1/test_frac) → один фолд = test;
    затем на остатке ещё раз → один фолд = val.
    """
    y = df["label"].values
    split = np.array(["train"] * len(df), dtype=object)

    k_test = max(2, round(1 / test_frac))
    sgk = StratifiedGroupKFold(n_splits=k_test, shuffle=True, random_state=seed)
    trval_idx, test_idx = next(sgk.split(np.zeros(len(y)), y, groups))
    split[test_idx] = "test"

    k_val = max(2, round((1 - test_frac) / val_frac))
    sgk2 = StratifiedGroupKFold(n_splits=k_val, shuffle=True, random_state=seed + 1)
    tr_rel, val_rel = next(sgk2.split(np.zeros(len(trval_idx)), y[trval_idx], groups[trval_idx]))
    split[trval_idx[val_rel]] = "val"
    return pd.Series(split, index=df.index, name="split")


def leakage_report(df: pd.DataFrame, pairs, split_col="split") -> dict:
    """Проверяем: сколько пар почти-дубликатов оказались бы по разные стороны сплита."""
    s = df[split_col].values
    cross = sum(1 for i, j, _ in pairs if s[i] != s[j])
    return dict(n_near_duplicate_pairs=len(pairs), cross_split_pairs=int(cross))


# ---------------------------------------------------------------------------
# 4. Dataset поверх кэша
# ---------------------------------------------------------------------------
def build_transforms(train: bool, size: int, cache_size: int, normalize: bool = True):
    """Аугментации только для train (никакой «статистики» по test/val не используется)."""
    ops = [T.ToImage(), T.ToDtype(torch.float32, scale=True)]
    if train:
        ops += [
            T.RandomResizedCrop(size, scale=(0.6, 1.0), antialias=True),
            T.RandomHorizontalFlip(),
            T.ColorJitter(0.2, 0.2, 0.2, 0.02),
        ]
    elif size != cache_size:
        ops += [T.Resize((size, size), antialias=True)]
    if normalize:
        ops += [T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return T.Compose(ops)


class PetsCache(Dataset):
    """
    Читает кадры из memmap. Можно подменить массив кадров (array=...) — так подаются
    варианты с заменённым фоном или водяным знаком, не трогая исходный кэш.
    """

    def __init__(self, cache_dir, indices, labels, size, train=False, cache_size=224, array=None,
                 post_aug_stamp=None, stamp_indices=()):
        """
        post_aug_stamp — функция uint8 (H,W,3) → uint8 (H,W,3), применяемая ПОСЛЕ аугментаций
        к кадрам из stamp_indices (индексы кэша). Нужна эксперименту с водяным знаком,
        где надпись не должна обрезаться случайным кропом (худший случай ярлыка).
        """
        self.cache_file = Path(cache_dir) / f"images_{cache_size}.npy"
        self.indices = np.asarray(indices)
        self.labels = np.asarray(labels)
        self.stamp = post_aug_stamp
        self.stamp_set = set(int(i) for i in stamp_indices)
        self.tf = build_transforms(train, size, cache_size, normalize=post_aug_stamp is None)
        self.norm = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        self._arr = array  # если None — открываем memmap лениво (безопасно для num_workers > 0)

    def __len__(self):
        return len(self.indices)

    def _array(self):
        if self._arr is None:
            self._arr = np.load(self.cache_file, mmap_mode="r")
        return self._arr

    def __getitem__(self, k):
        img = np.array(self._array()[self.indices[k]])  # копия: memmap открыт только на чтение
        x = self.tf(img)
        if self.stamp is not None:
            if int(self.indices[k]) in self.stamp_set:
                u8 = (x.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
                x = torch.from_numpy(self.stamp(u8)).permute(2, 0, 1).float() / 255.0
            x = self.norm(x)
        return x, int(self.labels[k])


def load_split_frame(cfg) -> pd.DataFrame:
    """Таблица splits.csv: image_id, label, class_name, species, split, group, + признаки качества."""
    return pd.read_csv(cfg["paths"]["splits_csv"])


def to_tensor_batch(arr_uint8: np.ndarray, size: int | None = None) -> torch.Tensor:
    """(N,H,W,3) uint8 → нормализованный тензор (N,3,H,W) для инференса."""
    x = torch.from_numpy(np.ascontiguousarray(arr_uint8)).permute(0, 3, 1, 2).float() / 255.0
    if size is not None and size != x.shape[-1]:
        x = torch.nn.functional.interpolate(x, size=(size, size), mode="bilinear", antialias=True, align_corners=False)
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (x - mean) / std
