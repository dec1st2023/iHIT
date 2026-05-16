import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class FlowPreprocessTests(unittest.TestCase):
    def test_resize_for_raft_uses_configured_square_size(self):
        from preprocess.flow import _resize_for_raft

        class FakeCV2:
            INTER_LINEAR = object()

            def __init__(self):
                self.calls = []

            def resize(self, image, size, interpolation=None):
                self.calls.append((image.shape, size, interpolation))
                return np.zeros((size[1], size[0], image.shape[2]), dtype=image.dtype)

        cv2 = FakeCV2()
        image = np.zeros((111, 111, 3), dtype=np.uint8)

        resized = _resize_for_raft(image, image_size=224, cv2=cv2)

        self.assertEqual(resized.shape, (224, 224, 3))
        self.assertEqual(cv2.calls, [((111, 111, 3), (224, 224), cv2.INTER_LINEAR)])

    def test_sanitize_flow_removes_non_finite_values(self):
        from preprocess.flow import _sanitize_flow

        flow = np.array([[[np.nan, np.inf], [-np.inf, 20000.0]]], dtype=np.float32)
        sanitized = _sanitize_flow(flow)

        self.assertTrue(np.isfinite(sanitized).all())
        self.assertEqual(float(sanitized[0, 0, 0]), 0.0)
        self.assertEqual(float(sanitized[0, 0, 1]), 0.0)
        self.assertEqual(float(sanitized[0, 1, 0]), 0.0)
        self.assertEqual(float(sanitized[0, 1, 1]), 10000.0)


if __name__ == "__main__":
    unittest.main()
