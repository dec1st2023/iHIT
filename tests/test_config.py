import sys
import tempfile
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class ConfigTests(unittest.TestCase):
    def test_load_config_resolves_minimal_schema_from_repo_root(self):
        from config import FLOW_FRAMES, IMAGE_SIZE, WINDOW_SIZE, load_config

        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.yaml"
            config_path.write_text(
                textwrap.dedent(
                    """
                    output_dir: runs
                    seed: 7
                    data:
                      root: data\\active
                    checkpoints:
                      stage1: checkpoint/stage1.pth
                      stage2: checkpoint/stage2.pth
                      raft: RAFT/model/raft-things.pth
                    training:
                      batch_size: 8
                      workers: 2
                      epochs: 3
                      lr: 0.0005
                      device: cpu
                      use_best_val_checkpoint: true
                    """
                ),
                encoding="utf-8",
            )

            config = load_config(config_path, repo_root=ROOT)

        self.assertEqual(config.mode, "all")
        self.assertEqual(config.stage, "full")
        self.assertEqual(config.experiment.output_dir, ROOT / "runs")
        self.assertEqual(config.experiment.seed, 7)
        self.assertEqual(config.data.root, ROOT / "data" / "active")
        self.assertEqual(config.data.raw_dir, ROOT / "data" / "active" / "raw")
        self.assertEqual(config.data.label_dir, ROOT / "data" / "active" / "label")
        self.assertEqual(config.data.flow_dir("test"), ROOT / "data" / "active" / "flow" / "test")
        self.assertEqual(config.data.position_dir("train"), ROOT / "data" / "active" / "position" / "train")
        self.assertEqual(config.raft_root, ROOT / "RAFT")
        self.assertEqual(config.checkpoints.stage2.name, "stage2.pth")
        self.assertEqual(config.training.batch_size, 8)
        self.assertTrue(config.training.use_best_val_checkpoint)
        self.assertEqual((WINDOW_SIZE, FLOW_FRAMES, IMAGE_SIZE), (20, 15, 224))
        self.assertFalse(hasattr(config.data, "window_size"))
        self.assertFalse(hasattr(config, "flow"))
        self.assertFalse(hasattr(config, "preprocess"))

    def test_load_config_defaults_to_github_reproduction_layout(self):
        from config import load_config

        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.yaml"
            config_path.write_text("", encoding="utf-8")
            config = load_config(config_path, repo_root=ROOT)

        self.assertEqual(config.experiment.output_dir, ROOT / "output")
        self.assertEqual(config.experiment.seed, 42)
        self.assertEqual(config.data.root, ROOT / "data")
        self.assertEqual(config.data.raw_dir, ROOT / "data" / "raw")
        self.assertEqual(config.data.label_dir, ROOT / "data" / "label")
        self.assertEqual(config.data.flow_dir("val"), ROOT / "data" / "flow" / "val")
        self.assertEqual(config.data.position_dir("test"), ROOT / "data" / "position" / "test")
        self.assertFalse(config.training.use_best_val_checkpoint)

    def test_load_config_reads_mode_and_stage(self):
        from config import load_config

        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.yaml"
            config_path.write_text("mode: evaluate\nstage: stage2\ntrain_stage: stage1\n", encoding="utf-8")
            config = load_config(config_path, repo_root=ROOT)

        self.assertEqual(config.mode, "evaluate")
        self.assertEqual(config.stage, "stage2")
        self.assertEqual(config.train_stage, "stage1")


if __name__ == "__main__":
    unittest.main()
