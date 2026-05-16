from __future__ import annotations

from pathlib import Path

from progress import progress

from .image_io import read_image, write_image
from .landmarks import crop_square, roi_detection


def generate_flow_dataset(input_dir: Path, output_dir: Path, model, flow_viz, input_padder_cls, window_size: int, sample_names=None, image_size: int = 224) -> None:
    cv2, mp, torch = _deps()
    output_dir.mkdir(parents=True, exist_ok=True)
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    folders = _sample_folders(input_dir, sample_names)
    for folder in progress(folders, desc="flow samples", unit="sample", total=len(folders)):
        frames = _frame_paths(folder)
        if len(frames) < 2:
            continue
        target_folder = output_dir / folder.name
        target_folder.mkdir(parents=True, exist_ok=True)
        left_roi, right_roi = roi_detection(frames[len(frames) // 2], face_mesh, window_size)
        for idx in range(1, len(frames)):
            prev_image = read_image(frames[idx - 1])
            curr_image = read_image(frames[idx])
            for suffix, roi in (("leye", left_roi), ("reye", right_roi)):
                prev_crop = crop_square(prev_image, roi)
                curr_crop = crop_square(curr_image, roi)
                write_image(target_folder / f"{idx}_{suffix}.jpg", curr_crop)
                flow_image = _estimate_flow_image(prev_crop, curr_crop, model, flow_viz, input_padder_cls, torch, image_size=image_size)
                write_image(target_folder / f"{idx}_{suffix}F.jpg", flow_image)


def _estimate_flow_image(first, second, model, flow_viz, input_padder_cls, torch, image_size: int = 224):
    cv2 = _cv2()
    device = next(model.parameters()).device
    first = _resize_for_raft(first, image_size=image_size, cv2=cv2)
    second = _resize_for_raft(second, image_size=image_size, cv2=cv2)
    image1 = _image_tensor(first, torch)
    image2 = _image_tensor(second, torch)
    padder = input_padder_cls(image1.shape)
    image1, image2 = padder.pad(image1, image2)
    image1 = image1.to(device)
    image2 = image2.to(device)
    with torch.no_grad():
        _, flow_up = model(image1, image2, iters=20, test_mode=True)
    flow = _sanitize_flow(flow_up[0].permute(1, 2, 0).cpu().numpy())
    return flow_viz.flow_to_image(flow)


def _resize_for_raft(image, image_size: int, cv2):
    if image_size <= 0:
        return image
    height, width = image.shape[:2]
    if height == image_size and width == image_size:
        return image
    return cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_LINEAR)


def _sanitize_flow(flow):
    import numpy as np

    finite = np.nan_to_num(flow, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(finite, -10000.0, 10000.0)


def _image_tensor(image, torch):
    import numpy as np

    array = np.asarray(image, dtype=np.uint8)
    return torch.from_numpy(array).permute(2, 0, 1).float()[None]


def _frame_paths(folder: Path) -> list[Path]:
    def key(path: Path) -> int:
        try:
            return int(path.stem)
        except ValueError:
            return 0

    return sorted(folder.glob("*.jpg"), key=key)


def _sample_folders(input_path: Path, sample_names) -> list[Path]:
    if sample_names is None:
        return sorted(path for path in input_path.iterdir() if path.is_dir())
    return [input_path / str(name) for name in sample_names if (input_path / str(name)).is_dir()]


def _deps():
    try:
        cv2 = _cv2()
        import mediapipe as mp
        import torch
    except Exception as exc:  # pragma: no cover - dependency message only
        raise RuntimeError("Flow preprocessing requires opencv-python, mediapipe, and torch.") from exc
    return cv2, mp, torch


def _cv2():
    import cv2

    return cv2
