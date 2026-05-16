import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class SequenceTests(unittest.TestCase):
    def test_process_positions_pads_short_sequences_to_thirty(self):
        from dataset.sequence import process_positions

        positions = np.arange(20 * 4, dtype=np.float32).reshape(20, 4)
        out = process_positions(positions, target_length=30)

        self.assertEqual(out.shape, (30, 4))
        np.testing.assert_array_equal(out[0], positions[0])
        np.testing.assert_array_equal(out[-1], positions[-1])

    def test_process_positions_trims_long_sequences_around_middle(self):
        from dataset.sequence import process_positions

        positions = np.arange(50 * 4, dtype=np.float32).reshape(50, 4)
        out = process_positions(positions, target_length=30)

        self.assertEqual(out.shape, (30, 4))
        np.testing.assert_array_equal(out[0], positions[10])
        np.testing.assert_array_equal(out[-1], positions[39])


if __name__ == "__main__":
    unittest.main()
