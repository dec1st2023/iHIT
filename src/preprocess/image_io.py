from __future__ import annotations

from pathlib import Path


def read_image(path: str | Path):
    import cv2
    import numpy as np

    image_path = Path(path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    data = np.fromfile(str(image_path), dtype=np.uint8)
    if data.size == 0:
        raise ValueError(f"Image is empty: {image_path}")
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not decode image: {image_path}")
    return image


def write_image(path: str | Path, image) -> None:
    import cv2

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    extension = output_path.suffix or ".jpg"
    ok, encoded = cv2.imencode(extension, image)
    if not ok:
        raise ValueError(f"Could not encode image for {output_path}")
    encoded.tofile(str(output_path))
