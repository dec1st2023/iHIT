import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


try:
    import torch
except Exception:
    torch = None


@unittest.skipIf(torch is None, "torch is not installed in this interpreter")
class ModelSmokeTests(unittest.TestCase):
    def test_stage2_lstm_forward_shape(self):
        from models.stage2_lstm import Stage2LSTM

        model = Stage2LSTM(hidden_size=8, num_layers=1)
        dirs = torch.eye(6)[[1, 4]]
        positions = torch.randn(2, 30, 4)

        output = model(dirs, positions)

        self.assertEqual(tuple(output.shape), (2, 2))

    def test_stage2_lstm_uses_three_independent_encoders(self):
        from models.stage2_lstm import Stage2LSTM

        model = Stage2LSTM(hidden_size=8, num_layers=1)

        self.assertIsNot(model.left_lstm, model.right_lstm)
        self.assertIsNot(model.left_lstm, model.double_lstm)
        self.assertIsNot(model.right_lstm, model.double_lstm)

    def test_stage1_flow_forward_shape(self):
        from models.stage1_r3d import Stage1FlowClassifier

        model = Stage1FlowClassifier(base_channels=2, layers=(1, 1, 1, 1))
        left = torch.randn(2, 15, 3, 32, 32)
        right = torch.randn(2, 15, 3, 32, 32)

        output = model(left, right)

        self.assertEqual(tuple(output.shape), (2, 6))


if __name__ == "__main__":
    unittest.main()
