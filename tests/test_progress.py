import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class ProgressTests(unittest.TestCase):
    def test_progress_uses_tqdm_with_description(self):
        from progress import progress

        calls = []

        def fake_tqdm(iterable, **kwargs):
            calls.append(kwargs)
            return iterable

        with patch("progress.tqdm", fake_tqdm):
            values = list(progress([1, 2, 3], desc="samples", unit="sample"))

        self.assertEqual(values, [1, 2, 3])
        self.assertEqual(calls, [{"desc": "samples", "unit": "sample", "total": None}])

    def test_progress_can_be_disabled(self):
        from progress import progress

        with patch("progress.tqdm", side_effect=AssertionError("should not call tqdm")):
            values = list(progress([1, 2], desc="disabled", disable=True))

        self.assertEqual(values, [1, 2])

    def test_evaluation_batches_use_progress_description(self):
        from evaluation import runners

        calls = []

        def fake_progress(iterable, **kwargs):
            calls.append(kwargs)
            return iterable

        with patch.object(runners, "progress", fake_progress, create=True):
            values = list(runners._progress_batches([1, 2], desc="stage1 test evaluation", total=2))

        self.assertEqual(values, [1, 2])
        self.assertEqual(calls[0]["desc"], "stage1 test evaluation")
        self.assertEqual(calls[0]["unit"], "batch")
        self.assertEqual(calls[0]["total"], 2)

    def test_stage1_mask_batches_use_progress_description(self):
        import stage1_masks

        calls = []

        def fake_progress(iterable, **kwargs):
            calls.append(kwargs)
            return iterable

        with patch.object(stage1_masks, "progress", fake_progress, create=True):
            values = list(stage1_masks._progress_batches([1, 2], desc="stage1 mask test", total=2))

        self.assertEqual(values, [1, 2])
        self.assertEqual(calls[0]["desc"], "stage1 mask test")
        self.assertEqual(calls[0]["unit"], "batch")
        self.assertEqual(calls[0]["total"], 2)


if __name__ == "__main__":
    unittest.main()
