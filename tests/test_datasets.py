import pickle
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


try:
    import torch  # noqa: F401
except Exception:
    torch = None


@unittest.skipIf(torch is None, "torch is not installed in this interpreter")
class DatasetTests(unittest.TestCase):
    def test_displacement_dataset_returns_stage2_sample(self):
        from dataset.datasets import DisplacementDataset

        with tempfile.TemporaryDirectory() as tmp:
            pos_dir = Path(tmp)
            names = np.array(["Z_LL_positive__g1"])
            left = {"Z_LL_positive__g1": [[i, i + 1] for i in range(16)]}
            right = {"Z_LL_positive__g1": [[i + 2, i + 3] for i in range(16)]}
            nose = {"Z_LL_positive__g1": [[i + 4, i + 5] for i in range(16)]}
            for filename, payload in [
                ("leye_positions.pkl", left),
                ("reye_positions.pkl", right),
                ("nose_positions.pkl", nose),
            ]:
                with open(pos_dir / filename, "wb") as handle:
                    pickle.dump(payload, handle)

            dataset = DisplacementDataset(
                positions_dir=pos_dir,
                names=names,
                dir_labels=np.eye(6)[[1]],
                posneg_labels=np.eye(2)[[1]],
                augment=False,
            )
            sample = dataset[0]

        self.assertEqual(tuple(sample.positions.shape), (30, 4))
        self.assertEqual(tuple(sample.y_dir.shape), (6,))
        self.assertEqual(tuple(sample.y_posneg.shape), (2,))


if __name__ == "__main__":
    unittest.main()
