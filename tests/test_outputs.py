import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class OutputTests(unittest.TestCase):
    def test_workflow_creates_timestamped_case_directory(self):
        from config import load_config
        from workflow import run

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "config.yaml"
            config_path.write_text(
                "mode: train\nstage: stage2\noutput_dir: output\n",
                encoding="utf-8",
            )
            config = load_config(config_path, repo_root=root)

            with mock.patch("workflow._timestamp", return_value="20260515_030405"), mock.patch("workflow.ensure_data_ready"), mock.patch("workflow.train") as train:
                train.return_value = {"stage2": root / "output" / "case_20260515_030405" / "checkpoints" / "stage2.pth"}
                result = run(config)

            case_dir = config.repo_root / "output" / "case_20260515_030405"
            train.assert_called_once()
            runtime_config = train.call_args.args[0]
            self.assertEqual(runtime_config.experiment.output_dir, case_dir)
            self.assertTrue(case_dir.is_dir())
            self.assertEqual(result.output_dir, case_dir)

    def test_training_writes_checkpoint_history_and_curve(self):
        from training.runners import _save_training_artifacts

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            history = [{"epoch": 1, "loss": 0.7, "accuracy": 0.5}, {"epoch": 2, "loss": 0.4, "accuracy": 0.75}]
            _save_training_artifacts(output_dir, "stage2", history)

            self.assertTrue((output_dir / "metrics" / "stage2_history.json").is_file())
            self.assertTrue((output_dir / "visuals" / "stage2_training_curve.png").is_file())

    def test_training_curve_series_contains_only_train_and_validation_accuracy(self):
        from training.runners import _accuracy_curve_series

        history = [
            {"epoch": 1, "train_loss": 0.7, "train_accuracy": 0.5, "val_loss": 0.8, "val_accuracy": 0.4},
            {"epoch": 2, "train_loss": 0.4, "train_accuracy": 0.75, "val_loss": 0.6, "val_accuracy": 0.65},
        ]

        epochs, series = _accuracy_curve_series(history)

        self.assertEqual(epochs, [1, 2])
        self.assertEqual(series, [("train accuracy", [0.5, 0.75]), ("validation accuracy", [0.4, 0.65])])

    def test_evaluation_writes_confusion_matrix_visual(self):
        from evaluation.metrics import classification_metrics
        from evaluation.runners import _save_confusion_visuals

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            metrics = {"stage2_vor": classification_metrics([0, 1, 1], [0, 1, 0], num_classes=2)}

            _save_confusion_visuals(output_dir, metrics)

            self.assertTrue((output_dir / "visuals" / "stage2_vor_conf_mat.png").is_file())

    def test_evaluation_writes_confusion_and_metric_summary_visuals(self):
        import numpy as np

        from evaluation.metrics import classification_metrics
        from evaluation.runners import _save_evaluation_visuals

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            metrics = {
                "stage1": classification_metrics([0, 1, 1, 2], [0, 1, 2, 2], num_classes=3),
                "stage2_vor": classification_metrics([0, 1, 1], [0, 1, 0], num_classes=2),
                "stage2_lvl": classification_metrics([0, 1, 1], [1, 1, 0], num_classes=2),
            }

            _save_evaluation_visuals(output_dir, metrics, prefix="full", paper_confusion=np.eye(12, dtype=np.int64))

            visuals = output_dir / "visuals"
            self.assertTrue((visuals / "stage1_conf_mat.png").is_file())
            self.assertTrue((visuals / "stage2_vor_conf_mat.png").is_file())
            self.assertTrue((visuals / "stage2_lvl_conf_mat.png").is_file())
            self.assertTrue((visuals / "stage2_conf_mat.png").is_file())
            self.assertTrue((visuals / "evaluation_metrics.png").is_file())

    def test_full_evaluation_uses_single_stage2_checkpoint_for_split_branches(self):
        import numpy as np

        from evaluation import runners
        from evaluation.metrics import classification_metrics

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            fresh_stage1 = output_dir / "checkpoints" / "stage1.pth"
            fresh_stage2 = output_dir / "checkpoints" / "stage2.pth"
            config = SimpleNamespace(
                experiment=SimpleNamespace(output_dir=output_dir),
                checkpoints=SimpleNamespace(stage1=Path("configured_stage1.pth"), stage2=Path("configured_stage2.pth")),
            )
            stage1_metrics = classification_metrics([0, 1], [0, 1], num_classes=2)
            stage2_result = runners.Stage2Evaluation(
                classification_metrics([0, 1], [0, 1], num_classes=2),
                np.eye(12, dtype=np.int64),
            )

            with mock.patch.object(runners, "_evaluate_stage1", return_value=stage1_metrics) as evaluate_stage1, mock.patch.object(runners, "_evaluate_stage2", return_value=stage2_result) as evaluate_stage2:
                metrics_path = runners.evaluate(config, stage="full", checkpoints={"stage1": fresh_stage1, "stage2": fresh_stage2})

            evaluate_stage1.assert_called_once_with(config, fresh_stage1)
            evaluate_stage2.assert_has_calls(
                [
                    mock.call(config, canal_group="vor", checkpoint_path=fresh_stage2, stage1_checkpoint_path=fresh_stage1),
                    mock.call(config, canal_group="lvl", checkpoint_path=fresh_stage2, stage1_checkpoint_path=fresh_stage1),
                ]
            )
            self.assertEqual(metrics_path.name, "full_metrics.json")

    def test_stage2_checkpoint_state_uses_single_model_payload(self):
        from evaluation.runners import _checkpoint_state

        checkpoint = {"model": {"w": "stage2"}}

        self.assertEqual(_checkpoint_state(checkpoint), {"w": "stage2"})

    def test_paper_confusion_matrix_uses_paper_label_order(self):
        from evaluation.runners import _stage2_confusion_matrix

        matrix = _stage2_confusion_matrix(true_canals=[0, 4], true_posneg=[1, 0], pred_posneg=[0, 1])

        self.assertEqual(matrix.shape, (12, 12))
        self.assertEqual(matrix[1, 0], 1)
        self.assertEqual(matrix[8, 9], 1)


if __name__ == "__main__":
    unittest.main()
