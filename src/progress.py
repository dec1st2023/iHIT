from __future__ import annotations

from collections.abc import Iterable
from typing import TypeVar


T = TypeVar("T")

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - dependency fallback only
    tqdm = None


def progress(iterable: Iterable[T], desc: str, unit: str = "it", total: int | None = None, disable: bool = False, **kwargs) -> Iterable[T]:
    if disable or tqdm is None:
        return iterable
    return tqdm(iterable, desc=desc, unit=unit, total=total, **kwargs)
