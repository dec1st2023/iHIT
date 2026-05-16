import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class SplitTests(unittest.TestCase):
    def test_load_split_validates_matching_shapes(self):
        from dataset.splits import load_split

        with tempfile.TemporaryDirectory() as tmp:
            split_dir = Path(tmp)
            np.save(split_dir / "test_names_vor.npy", np.array(["Z_LL_positive__g1", "Z_RL_negative__g2"]))
            np.save(split_dir / "test_dir_onehot_vor.npy", np.eye(6)[[1, 4]])
            np.save(split_dir / "test_posneg_onehot_vor.npy", np.eye(2)[[1, 0]])

            split = load_split(split_dir, subset="test", canal_group="vor")

        self.assertEqual(split.names.tolist(), ["Z_LL_positive__g1", "Z_RL_negative__g2"])
        self.assertEqual(split.dir_onehot.shape, (2, 6))
        self.assertEqual(split.posneg_onehot.shape, (2, 2))

    def test_load_split_rejects_mismatched_lengths(self):
        from dataset.splits import SplitShapeError, load_split

        with tempfile.TemporaryDirectory() as tmp:
            split_dir = Path(tmp)
            np.save(split_dir / "train_names_lvl.npy", np.array(["a", "b"]))
            np.save(split_dir / "train_dir_onehot_lvl.npy", np.eye(6)[[0]])
            np.save(split_dir / "train_posneg_onehot_lvl.npy", np.eye(2)[[0, 1]])

            with self.assertRaises(SplitShapeError):
                load_split(split_dir, subset="train", canal_group="lvl")


if __name__ == "__main__":
    unittest.main()
