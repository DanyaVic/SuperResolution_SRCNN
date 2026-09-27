"""Датасет для обучения SRCNN: случайные HR-патчи (канал Y) → бикубическая деградация «на лету»."""
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .imgproc import imresize


def augment(patch, rng):
    """Геометрические аугментации без потери информации: 8 вариантов (отражения × повороты на 90°)."""
    if rng.random() < 0.5:
        patch = patch[:, ::-1]
    if rng.random() < 0.5:
        patch = patch[::-1, :]
    return np.rot90(patch, rng.integers(4))


class SRPatchDataset(Dataset):
    """Читает нарезанные HR-фрагменты (N×S×S, uint8, канал Y) из memmap-файла prepare_data.py.

    Пара для обучения: HR-патч P×P → LR = bicubic↓s (с квантованием uint8) → вход сети = bicubic↑s(LR).
    Так сеть учится ровно на той деградации, которая используется на валидации/тесте.
    """

    def __init__(self, npy_path, scale, patch=96, length=None, augment=True, seed=0):
        self.path = str(npy_path)
        self.data = None  # memmap открывается лениво — корректно работает с num_workers > 0
        meta = json.loads(Path(npy_path).with_suffix(".json").read_text(encoding="utf-8"))
        self.n, self.size = meta["count"], meta["size"]
        self.scale, self.augment, self.seed = scale, augment, seed
        self.patch = patch - patch % scale
        assert self.patch <= self.size, "патч больше нарезанного фрагмента"
        self.length = length or self.n
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.length

    def _open(self):
        if self.data is None:
            self.data = np.load(self.path, mmap_mode="r")
            info = torch.utils.data.get_worker_info()
            self.rng = np.random.default_rng(self.seed + (info.id + 1) * 7919 if info else self.seed)

    def __getitem__(self, idx):
        self._open()
        rng, p, s = self.rng, self.patch, self.scale
        img = self.data[rng.integers(self.n)]
        y0, x0 = rng.integers(0, self.size - p + 1, size=2)
        hr = np.array(img[y0:y0 + p, x0:x0 + p])
        if self.augment:
            hr = np.ascontiguousarray(augment(hr, rng))
        lr = imresize(hr, (p // s, p // s))
        inp = imresize(lr, (p, p))
        to_t = lambda a: torch.from_numpy(a.astype(np.float32) / 255.0)[None]  # noqa: E731
        return to_t(inp), to_t(hr)
