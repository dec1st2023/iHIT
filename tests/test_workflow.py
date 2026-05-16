import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class WorkflowTests(unittest.TestCase):
    def test_train_mode_prepares_data_and_trains_selected_stage(self):
        from config import load_config
        from workflow import run

        config = load_config(_write_config("mode: train\nstage: full\ntrain_stage: stage1\n"), repo_root=ROOT)

        with mock.patch("workflow.ensure_data_ready") as ensure_data, mock.patch("workflow.train") as train, mock.patch("workflow.evaluate") as evaluate:
            train.return_value = {"stage1": ROOT / "runs" / "demo" / "stage1.pth"}
            result = run(config)

        ensure_data.assert_called_once()
        self.assertEqual(ensure_data.call_args.kwargs, {"include_training": True, "include_evaluation": False})
        runtime_config = train.call_args.args[0]
        self.assertEqual(runtime_config.stage, "stage1")
        self.assertIn("case_", runtime_config.experiment.output_dir.name)
        train.assert_called_once_with(runtime_config, stage="stage1")
        evaluate.assert_not_called()
        self.assertEqual(result.trained_checkpoints["stage1"].name, "stage1.pth")

    def test_train_stage_controls_all_mode_training_and_evaluation(self):
        from config import load_config
        from workflow import run

        fresh = {
            "stage2": ROOT / "runs" / "demo" / "stage2.pth",
        }
        config = load_config(_write_config("mode: all\nstage: full\ntrain_stage: stage2\n"), repo_root=ROOT)

        with mock.patch("workflow.ensure_data_ready") as ensure_data, mock.patch("workflow.train") as train, mock.patch("workflow.evaluate") as evaluate:
            train.return_value = fresh
            evaluate.return_value = ROOT / "runs" / "demo" / "stage2_metrics.json"
            run(config)

        runtime_config = train.call_args.args[0]
        self.assertEqual(runtime_config.stage, "stage2")
        ensure_data.assert_called_once_with(runtime_config, include_training=True, include_evaluation=True)
        train.assert_called_once_with(runtime_config, stage="stage2")
        evaluate.assert_called_once_with(runtime_config, stage="stage2", checkpoints=fresh)

    def test_evaluate_mode_prepares_data_and_uses_configured_checkpoints(self):
        from config import load_config
        from workflow import run

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stage1 = root / "checkpoint" / "stage1.pth"
            stage2 = root / "checkpoint" / "stage2.pth"
            stage1.parent.mkdir()
            stage1.write_bytes(b"stage1")
            stage2.write_bytes(b"stage2")
            config = load_config(
                _write_config(
                    """
mode: evaluate
stage: full
train_stage: stage2
checkpoints:
  stage1: checkpoint/stage1.pth
  stage2: checkpoint/stage2.pth
"""
                ),
                repo_root=root,
            )

            with mock.patch("workflow.ensure_data_ready") as ensure_data, mock.patch("workflow.train") as train, mock.patch("workflow.evaluate") as evaluate:
                evaluate.return_value = ROOT / "runs" / "demo" / "full_metrics.json"
                result = run(config)

            ensure_data.assert_called_once()
            self.assertEqual(ensure_data.call_args.kwargs, {"include_training": False, "include_evaluation": True})
            runtime_config = evaluate.call_args.args[0]
            self.assertEqual(runtime_config.stage, "full")
            self.assertIn("case_", runtime_config.experiment.output_dir.name)
            train.assert_not_called()
            evaluate.assert_called_once_with(runtime_config, stage="full", checkpoints=None)
            self.assertEqual(result.metrics_path.name, "full_metrics.json")

    def test_evaluate_mode_reports_missing_configured_checkpoint_before_evaluation(self):
        from config import load_config
        from workflow import run

        config = load_config(
            _write_config(
                """
mode: evaluate
stage: stage1
checkpoints:
  stage1: missing/stage1.pth
"""
            ),
            repo_root=ROOT,
        )

        with mock.patch("workflow.ensure_data_ready"), mock.patch("workflow.evaluate") as evaluate:
            with self.assertRaisesRegex(FileNotFoundError, "stage1 checkpoint not found"):
                run(config)

        evaluate.assert_not_called()

    def test_evaluate_stage2_requires_stage1_checkpoint_for_mask(self):
        from config import load_config
        from workflow import run

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stage2 = root / "checkpoint" / "stage2.pth"
            stage2.parent.mkdir()
            stage2.write_bytes(b"stage2")
            config = load_config(
                _write_config(
                    """
mode: evaluate
stage: stage2
checkpoints:
  stage2: checkpoint/stage2.pth
"""
                ),
                repo_root=root,
            )

            with mock.patch("workflow.ensure_data_ready"), mock.patch("workflow.evaluate") as evaluate:
                with self.assertRaisesRegex(FileNotFoundError, "stage1 checkpoint not found"):
                    run(config)

            evaluate.assert_not_called()

    def test_all_mode_passes_fresh_training_checkpoints_to_evaluation(self):
        from config import load_config
        from workflow import run

        fresh = {
            "stage1": ROOT / "runs" / "demo" / "stage1.pth",
            "stage2": ROOT / "runs" / "demo" / "stage2.pth",
        }
        config = load_config(_write_config("mode: all\nstage: full\n"), repo_root=ROOT)

        with mock.patch("workflow.ensure_data_ready") as ensure_data, mock.patch("workflow.train") as train, mock.patch("workflow.evaluate") as evaluate:
            train.return_value = fresh
            evaluate.return_value = ROOT / "runs" / "demo" / "full_metrics.json"
            result = run(config)

        ensure_data.assert_called_once()
        self.assertEqual(ensure_data.call_args.kwargs, {"include_training": True, "include_evaluation": True})
        runtime_config = train.call_args.args[0]
        self.assertEqual(runtime_config.stage, "full")
        self.assertIn("case_", runtime_config.experiment.output_dir.name)
        train.assert_called_once_with(runtime_config, stage="full")
        evaluate.assert_called_once_with(runtime_config, stage="full", checkpoints=fresh)
        self.assertEqual(result.trained_checkpoints, fresh)


class DataSetupTests(unittest.TestCase):
    def test_data_setup_skips_rebuild_when_required_assets_exist(self):
        from config import load_config
        from data_setup import ensure_data_ready

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_required_assets(root, include_train=True, include_eval=True)
            config = load_config(_write_config("mode: all\nstage: full\n"), repo_root=root)

            with mock.patch("data_setup.execute_rebuild") as execute_rebuild:
                ensure_data_ready(config, include_training=True, include_evaluation=True)

        execute_rebuild.assert_not_called()

    def test_data_setup_rebuilds_when_required_assets_are_missing(self):
        from config import load_config
        from data_setup import ensure_data_ready

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = load_config(_write_config("mode: train\nstage: full\n"), repo_root=root)

            with mock.patch("data_setup.build_rebuild_plan") as build_plan, mock.patch("data_setup.execute_rebuild") as execute_rebuild:
                build_plan.return_value = object()
                ensure_data_ready(config, include_training=True, include_evaluation=False)

        build_plan.assert_called_once_with(config.repo_root, window_size=20, seed=42)
        execute_rebuild.assert_called_once_with(build_plan.return_value, execute=True, generate=True, config=config)

    def test_data_setup_rebuilds_when_validation_assets_are_missing_for_training(self):
        from config import load_config
        from data_setup import ensure_data_ready

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_required_assets(root, include_train=True, include_val=False, include_eval=False)
            config = load_config(_write_config("mode: train\nstage: full\n"), repo_root=root)

            with mock.patch("data_setup.build_rebuild_plan") as build_plan, mock.patch("data_setup.execute_rebuild") as execute_rebuild:
                build_plan.return_value = object()
                ensure_data_ready(config, include_training=True, include_evaluation=False)

        build_plan.assert_called_once()
        execute_rebuild.assert_called_once()


def _write_config(body: str) -> Path:
    tmp = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8")
    with tmp:
        tmp.write(body)
    return Path(tmp.name)


def _write_required_assets(root: Path, include_train: bool, include_eval: bool, include_val: bool = True) -> None:
    label = root / "data" / "label"
    label.mkdir(parents=True)
    for subset in ("train", "val", "test"):
        for suffix in ("", "_vor", "_lvl"):
            for kind in ("names", "dir_onehot", "posneg_onehot"):
                (label / f"{subset}_{kind}{suffix}.npy").write_bytes(b"present")
    raw = root / "data" / "raw" / "sample"
    raw.mkdir(parents=True)
    if include_train:
        (root / "data" / "flow" / "train" / "sample").mkdir(parents=True)
        train_pos = root / "data" / "position" / "train"
        train_pos.mkdir(parents=True)
        for name in ("leye_positions.pkl", "reye_positions.pkl", "nose_positions.pkl"):
            (train_pos / name).write_bytes(b"present")
    if include_val:
        (root / "data" / "flow" / "val" / "sample").mkdir(parents=True)
        val_pos = root / "data" / "position" / "val"
        val_pos.mkdir(parents=True)
        for name in ("leye_positions.pkl", "reye_positions.pkl", "nose_positions.pkl"):
            (val_pos / name).write_bytes(b"present")
    if include_eval:
        (root / "data" / "flow" / "test" / "sample").mkdir(parents=True)
        eval_pos = root / "data" / "position" / "test"
        eval_pos.mkdir(parents=True)
        for name in ("leye_positions.pkl", "reye_positions.pkl", "nose_positions.pkl"):
            (eval_pos / name).write_bytes(b"present")


if __name__ == "__main__":
    unittest.main()
