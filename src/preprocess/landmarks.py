from __future__ import annotations

import pickle
from pathlib import Path

from progress import progress

from .image_io import read_image


EYE_LANDMARKS = {
    "left": (33, 133, 159, 145),
    "right": (362, 263, 386, 374),
}


def process_eyes(input_dir: str | Path, output_dir: str | Path, window_size: int = 80, sample_names=None) -> None:
    cv2, mp = _vision_deps()
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    left_positions = {}
    right_positions = {}
    folders = _sample_folders(input_path, sample_names)
    for folder in progress(folders, desc="eye landmarks", unit="sample", total=len(folders)):
        frame_paths = _frame_paths(folder)
        if not frame_paths:
            continue
        left_roi, right_roi = roi_detection(frame_paths[len(frame_paths) // 2], face_mesh, window_size)
        left_points = []
        right_points = []
        for frame_path in frame_paths:
            image = read_image(frame_path)
            left_points = eye_detection(crop_square(image, left_roi), left_points)
            right_points = eye_detection(crop_square(image, right_roi), right_points)
        left_positions[folder.name] = left_points
        right_positions[folder.name] = right_points

    _dump_pickle(output_path / "leye_positions.pkl", left_positions)
    _dump_pickle(output_path / "reye_positions.pkl", right_positions)


def process_nose(input_dir: str | Path, output_dir: str | Path, sample_names=None) -> None:
    _, _, modelscope_deps = _deps()
    pipeline, tasks = modelscope_deps
    face_detection1 = pipeline(tasks.face_detection, "damo/cv_resnet50_face-detection_retinaface")
    face_detection2 = _optional_scrfd_detector(pipeline, tasks)
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    nose_positions = {}
    folders = _sample_folders(input_path, sample_names)
    for folder in progress(folders, desc="nose landmarks", unit="sample", total=len(folders)):
        points = []
        for frame_path in _frame_paths(folder):
            x, y = _average_nose_keypoint(face_detection1(str(frame_path)), face_detection2(str(frame_path)), frame_path)
            points.append([x, y])
        nose_positions[folder.name] = points
    _dump_pickle(output_path / "nose_positions.pkl", nose_positions)


def _optional_scrfd_detector(pipeline, tasks):
    try:
        return pipeline(task=tasks.face_detection, model="damo/cv_resnet_facedetection_scrfd10gkps")
    except Exception as standard_exc:
        try:
            return _build_windows_scrfd_detector()
        except Exception as windows_exc:
            raise RuntimeError(
                "Windows ModelScope SCRFD is required for strict iHIT reproduction. "
                f"Standard ModelScope failed with: {standard_exc}. "
                f"Windows adapter failed with: {windows_exc}."
            ) from windows_exc


def _build_windows_scrfd_detector():
    from preprocess.modelscope_scrfd_windows import build_detector

    return build_detector()


def _average_nose_keypoint(retinaface_result, scrfd_result, frame_path: Path) -> tuple[float, float]:
    x1, y1 = _nose_keypoint(retinaface_result, frame_path)
    x2, y2 = _nose_keypoint(scrfd_result, frame_path)
    return (x1 + x2) / 2, (y1 + y2) / 2


def _nose_keypoint(result, frame_path: Path) -> tuple[float, float]:
    keypoints = result.get("keypoints") if isinstance(result, dict) else None
    if not keypoints:
        raise RuntimeError(f"No face keypoints detected in {frame_path}")
    point = keypoints[0]
    return float(point[4]), float(point[5])


def roi_detection(image_path: str | Path, face_mesh, window_size: int):
    cv2, _ = _vision_deps()
    image = read_image(image_path)
    result = face_mesh.process(image)
    if not result.multi_face_landmarks:
        raise RuntimeError(f"No face landmarks detected in {image_path}")
    face = result.multi_face_landmarks[0]
    height, width = image.shape[:2]

    def box(indices):
        left_idx, right_idx, top_idx, bottom_idx = indices
        x_min = int(face.landmark[left_idx].x * width - window_size)
        x_max = int(face.landmark[right_idx].x * width + window_size)
        y_min = int(face.landmark[top_idx].y * height - window_size)
        y_max = int(face.landmark[bottom_idx].y * height + window_size)
        return [y_min, y_max, x_min, x_max]

    return box(EYE_LANDMARKS["left"]), box(EYE_LANDMARKS["right"])


def eye_detection(roi, positions):
    cv2, _ = _vision_deps()
    import numpy as np

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (1, 1), 0)
    _, threshold = cv2.threshold(blur, 25, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((7, 7), np.uint8)
    closing = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda item: cv2.contourArea(item), reverse=True)
    x = y = w = h = 0
    for contour in contours:
        (_, _), radius = cv2.minEnclosingCircle(contour)
        if radius * 2 > 100:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        break
    positions.append([x + w / 2, y + h / 2])
    return positions


def crop_square(image, coordinates):
    import numpy as np

    y1, y2, x1, x2 = coordinates
    height, width = image.shape[:2]
    if x2 > width:
        x2 = width
    if y2 > height:
        y2 = height
    h = y2 - y1
    w = x2 - x1
    side = int(np.maximum(h, w))
    y = y1 + (h - side) // 2
    x = x1 + (w - side) // 2
    if x < 0:
        x = 0
    if y < 0:
        y = 0
    if x + side > width:
        side = width - x
    if y + side > height:
        side = height - y
    return image[y : y + side, x : x + side]


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


def _dump_pickle(path: Path, payload) -> None:
    with open(path, "wb") as handle:
        pickle.dump(payload, handle)


def _deps():
    try:
        cv2, mp = _vision_deps()
        from modelscope.pipelines import pipeline
        from modelscope.utils.constant import Tasks
    except Exception as exc:  # pragma: no cover - dependency message only
        raise RuntimeError("Landmark preprocessing requires opencv-python, mediapipe, and modelscope.") from exc
    return cv2, mp, (pipeline, Tasks)


def _vision_deps():
    try:
        import cv2
        import mediapipe as mp
    except Exception as exc:  # pragma: no cover - dependency message only
        raise RuntimeError("Landmark preprocessing requires opencv-python and mediapipe.") from exc
    return cv2, mp
