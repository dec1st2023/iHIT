from __future__ import annotations

import pickle
import re
from collections import namedtuple
from pathlib import Path

import numpy as np
from PIL import Image

from .sequence import process_positions


DisplacementSample = namedtuple("DisplacementSample", ["positions", "y_posneg", "y_dir", "name"])
FlowSample = namedtuple("FlowSample", ["left_flow", "right_flow", "y_dir", "y_posneg", "name"])


class DisplacementDataset:
    """Dataset for FloPNet stage 2 head/eye displacement classification."""

    def __init__(self, positions_dir, names, dir_labels, posneg_labels, augment: bool = False, noise_rate: float = 0.3):
        self.positions_dir = Path(positions_dir)
        self.names = np.asarray(names)
        self.dir_labels = np.asarray(dir_labels)
        self.posneg_labels = np.asarray(posneg_labels)
        self.augment = augment
        self.noise_rate = noise_rate
        self._left = _load_pickle(self.positions_dir / "leye_positions.pkl")
        self._right = _load_pickle(self.positions_dir / "reye_positions.pkl")
        self._nose = _load_pickle(self.positions_dir / "nose_positions.pkl")

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, index: int):
        torch = _torch()
        name = str(self.names[index])
        positions = self._positions_for(name)
        if self.augment:
            positions = _inject_noise(positions, self.noise_rate)
        return DisplacementSample(
            positions=torch.as_tensor(positions, dtype=torch.float32),
            y_posneg=torch.as_tensor(self.posneg_labels[index], dtype=torch.float32),
            y_dir=torch.as_tensor(self.dir_labels[index], dtype=torch.float32),
            name=name,
        )

    def _positions_for(self, name: str) -> np.ndarray:
        left = _delta(self._left[name], clamp=35)
        right = _delta(self._right[name], clamp=35)
        nose = _delta(self._nose[name], clamp=None)
        eyes = np.asarray(left + right, dtype=np.float32)
        head = np.asarray(nose + nose, dtype=np.float32)
        return process_positions(np.concatenate([eyes, head], axis=1), target_length=30)


class FlowDataset:
    """Dataset for FloPNet stage 1 SCC identification from eye optical flow images."""

    def __init__(self, flow_dir, names, dir_labels, posneg_labels, frames: int = 15, image_size: int = 224):
        self.flow_dir = Path(flow_dir)
        self.names = np.asarray(names)
        self.dir_labels = np.asarray(dir_labels)
        self.posneg_labels = np.asarray(posneg_labels)
        self.frames = frames
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, index: int):
        torch = _torch()
        name = str(self.names[index])
        folder = self.flow_dir / name
        left = _load_eye_sequence(folder, "leyeF", self.frames, self.image_size)
        right = _load_eye_sequence(folder, "reyeF", self.frames, self.image_size)
        return FlowSample(
            left_flow=torch.as_tensor(left, dtype=torch.float32),
            right_flow=torch.as_tensor(right, dtype=torch.float32),
            y_dir=torch.as_tensor(self.dir_labels[index], dtype=torch.float32),
            y_posneg=torch.as_tensor(self.posneg_labels[index], dtype=torch.float32),
            name=name,
        )


def _load_pickle(path: Path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def _delta(points, clamp: int | None) -> list[list[float]]:
    array = np.asarray(points, dtype=np.float32)
    out = [[0.0, 0.0]]
    for index in range(1, len(array)):
        dx, dy = (array[index] - array[index - 1]).tolist()
        if clamp is not None:
            if abs(dx) > clamp:
                dx = 0.0
            if abs(dy) > clamp:
                dy = 0.0
        out.append([dx, dy])
    return out


def _inject_noise(data: np.ndarray, noise_rate: float) -> np.ndarray:
    out = data.copy()
    for column in range(out.shape[1]):
        std = np.std(out[:, column])
        if std == 0:
            continue
        pct = np.random.uniform(0.05, noise_rate)
        out[:, column] += np.random.normal(0.0, pct * std, size=out.shape[0])
    return out


def _load_eye_sequence(folder: Path, suffix: str, frames: int, image_size: int) -> np.ndarray:
    files = sorted(
        folder.glob(f"*_{suffix}.jpg"),
        key=lambda path: int(re.match(r"(\d+)_", path.name).group(1)) if re.match(r"(\d+)_", path.name) else 0,
    )
    if not files:
        raise FileNotFoundError(f"No {suffix} images found in {folder}")
    selected = _select_center(files, frames)
    arrays = []
    for path in selected:
        image = Image.open(path).convert("RGB").resize((image_size, image_size))
        array = np.asarray(image, dtype=np.float32) / 255.0
        arrays.append(np.transpose(array, (2, 0, 1)))
    return np.stack(arrays, axis=0)


def _select_center(items: list[Path], length: int) -> list[Path]:
    if len(items) == length:
        return items
    if len(items) > length:
        start = (len(items) - length) // 2
        return items[start : start + length]
    front = [items[0]] * ((length - len(items)) // 2)
    back = [items[-1]] * (length - len(items) - len(front))
    return front + items + back


def _torch():
    try:
        import torch
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("PyTorch is required for iHIT datasets.") from exc
    return torch
