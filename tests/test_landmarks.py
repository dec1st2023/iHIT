import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class LandmarkTests(unittest.TestCase):
    def test_scrfd_detector_falls_back_to_windows_modelscope_adapter(self):
        import preprocess.landmarks as landmarks

        def pipeline(*args, **kwargs):
            raise ModuleNotFoundError("No module named 'mmcv._ext'")

        class Tasks:
            face_detection = "face-detection"

        detector = object()
        original_builder = landmarks._build_windows_scrfd_detector
        try:
            landmarks._build_windows_scrfd_detector = lambda: detector
            self.assertIs(landmarks._optional_scrfd_detector(pipeline, Tasks), detector)
        finally:
            landmarks._build_windows_scrfd_detector = original_builder

    def test_scrfd_detector_reports_standard_and_windows_failures(self):
        import preprocess.landmarks as landmarks

        def pipeline(*args, **kwargs):
            raise ModuleNotFoundError("No module named 'mmcv._ext'")

        class Tasks:
            face_detection = "face-detection"

        original_builder = landmarks._build_windows_scrfd_detector
        try:
            landmarks._build_windows_scrfd_detector = lambda: (_ for _ in ()).throw(RuntimeError("missing torch"))
            with self.assertRaisesRegex(RuntimeError, "Windows ModelScope SCRFD"):
                landmarks._optional_scrfd_detector(pipeline, Tasks)
        finally:
            landmarks._build_windows_scrfd_detector = original_builder

    def test_nose_keypoint_reads_third_landmark_pair(self):
        from preprocess.landmarks import _nose_keypoint

        x, y = _nose_keypoint({"keypoints": [[1, 2, 3, 4, 5.5, 6.5, 7, 8, 9, 10]]}, Path("frame.jpg"))

        self.assertEqual((x, y), (5.5, 6.5))

    def test_nose_keypoint_matches_original_retinaface_scrfd_average(self):
        from preprocess.landmarks import _average_nose_keypoint

        retinaface = {"keypoints": [[0, 0, 0, 0, 4.0, 8.0, 0, 0, 0, 0]]}
        scrfd = {"keypoints": [[0, 0, 0, 0, 6.0, 10.0, 0, 0, 0, 0]]}

        self.assertEqual(_average_nose_keypoint(retinaface, scrfd, Path("frame.jpg")), (5.0, 9.0))

    def test_crop_square_preserves_original_roi_geometry_before_clamping(self):
        from preprocess.landmarks import crop_square

        image = np.zeros((100, 100, 3), dtype=np.uint8)
        crop = crop_square(image, [-20, 40, -10, 50])

        self.assertEqual(crop.shape[:2], (60, 60))

    def test_roi_detection_passes_image_to_facemesh_like_original_script(self):
        import preprocess.landmarks as landmarks

        image = np.zeros((100, 100, 3), dtype=np.uint8)

        class FakeCv2:
            COLOR_BGR2RGB = 1

            def cvtColor(self, payload, code):
                return "converted"

        class FakePoint:
            def __init__(self, x, y):
                self.x = x
                self.y = y

        class FakeFace:
            landmark = [FakePoint(0.5, 0.5) for _ in range(400)]

        class FakeResult:
            multi_face_landmarks = [FakeFace()]

        class FakeFaceMesh:
            def __init__(self):
                self.payload = None

            def process(self, payload):
                self.payload = payload
                return FakeResult()

        fake_mesh = FakeFaceMesh()
        original_deps = landmarks._vision_deps
        original_read_image = landmarks.read_image
        try:
            landmarks._vision_deps = lambda: (FakeCv2(), None)
            landmarks.read_image = lambda path: image
            landmarks.roi_detection(Path("0.jpg"), fake_mesh, 20)
        finally:
            landmarks._vision_deps = original_deps
            landmarks.read_image = original_read_image

        self.assertIs(fake_mesh.payload, image)

if __name__ == "__main__":
    unittest.main()
