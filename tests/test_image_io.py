import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class ImageIOTests(unittest.TestCase):
    def test_read_and_write_image_support_unicode_windows_paths(self):
        try:
            import cv2  # noqa: F401
        except Exception:
            self.skipTest("opencv-python is not installed in this interpreter")
        from preprocess.image_io import read_image, write_image

        with tempfile.TemporaryDirectory(prefix="璧勬枡_") as tmp:
            path = Path(tmp) / "鏍锋湰" / "0.jpg"
            image = np.full((8, 8, 3), 127, dtype=np.uint8)
            write_image(path, image)
            loaded = read_image(path)

        self.assertEqual(loaded.shape, (8, 8, 3))
        self.assertEqual(loaded.dtype, np.uint8)


if __name__ == "__main__":
    unittest.main()
