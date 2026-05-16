from __future__ import annotations

import numpy as np


def process_positions(positions, target_length: int = 30) -> np.ndarray:
    """Pad or center-trim a position sequence to a fixed frame count."""

    array = np.asarray(positions, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"positions must be 2D, got shape {array.shape}")
    if len(array) == 0:
        raise ValueError("positions must contain at least one frame")
    if target_length <= 0:
        raise ValueError("target_length must be positive")

    length = len(array)
    if length == target_length:
        return array
    if length < target_length:
        front_pad = (target_length - length) // 2
        back_pad = target_length - length - front_pad
        front = np.repeat(array[:1], front_pad, axis=0)
        back = np.repeat(array[-1:], back_pad, axis=0)
        return np.concatenate([front, array, back], axis=0)

    start = (length - target_length) // 2
    end = start + target_length
    return array[start:end]


def head_tail_frames(frames, target_length: int):
    """Pad or center-trim a tensor-like frame sequence."""

    length = len(frames)
    if length == target_length:
        return frames
    if length > target_length:
        start = (length - target_length) // 2
        return frames[start : start + target_length]

    import torch

    first = frames[0:1].repeat((target_length - length) // 2, *([1] * (frames.ndim - 1)))
    back_count = target_length - length - len(first)
    last = frames[-1:].repeat(back_count, *([1] * (frames.ndim - 1)))
    return torch.cat([first, frames, last], dim=0)
