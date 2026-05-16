from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


class SplitShapeError(ValueError):
    """Raised when split metadata arrays do not align."""


@dataclass(frozen=True)
class Split:
    subset: str
    canal_group: str
    names: np.ndarray
    dir_onehot: np.ndarray
    posneg_onehot: np.ndarray


def load_split(split_dir: str | Path, subset: str, canal_group: str = "all") -> Split:
    split_path = Path(split_dir)
    suffix = "" if canal_group in {"all", "full"} else f"_{canal_group}"
    names = np.load(split_path / f"{subset}_names{suffix}.npy", allow_pickle=True)
    dir_onehot = np.load(split_path / f"{subset}_dir_onehot{suffix}.npy", allow_pickle=True)
    posneg_onehot = np.load(split_path / f"{subset}_posneg_onehot{suffix}.npy", allow_pickle=True)

    if not (len(names) == len(dir_onehot) == len(posneg_onehot)):
        raise SplitShapeError(
            f"{subset}/{canal_group} arrays have mismatched lengths: "
            f"names={len(names)}, dir={len(dir_onehot)}, posneg={len(posneg_onehot)}"
        )
    if dir_onehot.ndim != 2 or dir_onehot.shape[1] != 6:
        raise SplitShapeError(f"direction labels must have shape (N, 6), got {dir_onehot.shape}")
    if posneg_onehot.ndim != 2 or posneg_onehot.shape[1] != 2:
        raise SplitShapeError(f"posneg labels must have shape (N, 2), got {posneg_onehot.shape}")

    return Split(subset=subset, canal_group=canal_group, names=names, dir_onehot=dir_onehot, posneg_onehot=posneg_onehot)
