import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class RaftAdapterTests(unittest.TestCase):
    def test_raft_args_support_attribute_and_membership_access(self):
        from flow.raft_adapter import _RAFTArgs

        args = _RAFTArgs()
        self.assertTrue(args.small is False)
        self.assertIn("small", args)
        self.assertNotIn("dropout", args)
        args.dropout = 0
        self.assertIn("dropout", args)

    def test_strip_module_prefix_from_dataparallel_checkpoint(self):
        from flow.raft_adapter import _strip_module_prefix

        state = {"module.encoder.weight": 1, "module.encoder.bias": 2}
        self.assertEqual(_strip_module_prefix(state), {"encoder.weight": 1, "encoder.bias": 2})


if __name__ == "__main__":
    unittest.main()
