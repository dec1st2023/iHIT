import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class TrainingRuntimeTests(unittest.TestCase):
    def test_auto_device_uses_cuda_when_available(self):
        from training.runners import _device_name

        class Cuda:
            @staticmethod
            def is_available():
                return True

        class Torch:
            cuda = Cuda()

        self.assertEqual(_device_name(Torch, "auto"), "cuda")

    def test_auto_device_falls_back_to_cpu(self):
        from training.runners import _device_name

        class Cuda:
            @staticmethod
            def is_available():
                return False

        class Torch:
            cuda = Cuda()

        self.assertEqual(_device_name(Torch, "auto"), "cpu")

    def test_progress_batches_uses_tqdm_description(self):
        from training import runners

        calls = {}

        def fake_tqdm(iterable, **kwargs):
            calls.update(kwargs)
            return iterable

        with mock.patch.object(runners, "tqdm", fake_tqdm, create=True):
            values = list(runners._progress_batches([1, 2], total=2, desc="stage1 epoch 1/80"))

        self.assertEqual(values, [1, 2])
        self.assertEqual(calls["total"], 2)
        self.assertEqual(calls["desc"], "stage1 epoch 1/80")
        self.assertFalse(calls["leave"])
        self.assertTrue(calls["dynamic_ncols"])

    def test_train_stage2_trains_stage1_first_when_checkpoint_is_missing(self):
        from training import runners

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            config = SimpleNamespace(experiment=SimpleNamespace(output_dir=output_dir), checkpoints=SimpleNamespace(stage1=None))
            stage1 = output_dir / "checkpoints" / "stage1.pth"
            stage2 = output_dir / "checkpoints" / "stage2.pth"

            with mock.patch.object(runners, "_train_stage1", return_value=stage1) as train_stage1, mock.patch.object(runners, "_train_stage2_bundle", return_value=stage2) as train_stage2:
                outputs = runners.train(config, stage="stage2")

            train_stage1.assert_called_once_with(config)
            train_stage2.assert_called_once_with(config, stage1_checkpoint=stage1)
            self.assertEqual(outputs, {"stage1": stage1, "stage2": stage2})

    def test_train_stage2_uses_existing_stage1_checkpoint(self):
        from training import runners

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            stage1 = output_dir / "configured" / "stage1.pth"
            stage1.parent.mkdir()
            stage1.write_bytes(b"stage1")
            stage2 = output_dir / "checkpoints" / "stage2.pth"
            config = SimpleNamespace(experiment=SimpleNamespace(output_dir=output_dir), checkpoints=SimpleNamespace(stage1=stage1))

            with mock.patch.object(runners, "_train_stage1") as train_stage1, mock.patch.object(runners, "_train_stage2_bundle", return_value=stage2) as train_stage2:
                outputs = runners.train(config, stage="stage2")

            train_stage1.assert_not_called()
            train_stage2.assert_called_once_with(config, stage1_checkpoint=stage1)
            self.assertEqual(outputs, {"stage2": stage2})

    def test_stage2_phase_uses_true_direction_labels_for_training(self):
        from training import runners

        model = mock.Mock()
        config = SimpleNamespace(training=SimpleNamespace(lr=0.001, epochs=1, use_best_val_checkpoint=False), experiment=SimpleNamespace(output_dir=ROOT / "runs" / "demo"))

        class Loader(list):
            dataset = [object()]

        train_loader = Loader([object()])
        val_loader = Loader([object()])

        class Optim:
            def __init__(self, parameters, lr):
                self.parameters = parameters
                self.lr = lr

        class Torch:
            class optim:
                Adam = Optim

        with mock.patch.object(runners, "_torch", return_value=Torch), mock.patch.object(runners, "_set_stage2_trainable"), mock.patch.object(runners, "_trainable_parameters", return_value=[object()]), mock.patch.object(runners, "_stage2_loader", side_effect=[train_loader, val_loader]), mock.patch.object(runners, "stage1_direction_lookup", create=True) as lookup, mock.patch.object(runners, "_classification_loop", return_value=runners.TrainingRun(history=[])) as loop, mock.patch.object(runners, "_save_training_artifacts"):
            runners._train_stage2_phase(config, model, ROOT / "stage1.pth", canal_group="vor", branches="double", name="stage2_vor", device="cpu")

        lookup.assert_not_called()
        self.assertIsNone(loop.call_args.kwargs["direction_lookup"])
        self.assertIsNone(loop.call_args.kwargs["val_direction_lookup"])

    def test_stage1_saves_best_validation_state_when_enabled(self):
        from training import runners

        saved = {}

        class Model:
            def to(self, device):
                return self

            def parameters(self):
                return [object()]

            def load_state_dict(self, state):
                self.loaded_state = state

            def state_dict(self):
                return getattr(self, "loaded_state", {"final": True})

        class Torch:
            class optim:
                class Adam:
                    def __init__(self, parameters, lr):
                        pass

            @staticmethod
            def device(name):
                return name

            @staticmethod
            def save(payload, path):
                saved[path] = payload
                path.write_bytes(b"checkpoint")

        class Loader(list):
            dataset = [object()]

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            model = Model()
            config = SimpleNamespace(
                training=SimpleNamespace(lr=0.001, epochs=1, device="cpu", use_best_val_checkpoint=True),
                experiment=SimpleNamespace(output_dir=output_dir),
            )
            fake_module = types.SimpleNamespace(Stage1FlowClassifier=lambda: model)
            run = runners.TrainingRun(history=[{"epoch": 1, "val_accuracy": 0.9}], best_state={"best": True}, best_epoch=1, best_val_accuracy=0.9)

            with mock.patch.dict(sys.modules, {"models.stage1_r3d": fake_module}), mock.patch.object(runners, "_torch", return_value=Torch), mock.patch.object(runners, "_stage1_loader", return_value=Loader()), mock.patch.object(runners, "_classification_loop", return_value=run), mock.patch.object(runners, "_save_training_artifacts"):
                checkpoint = runners._train_stage1(config)

        self.assertEqual(saved[checkpoint]["model"], {"best": True})

    def test_stage1_saves_final_state_when_best_validation_is_disabled(self):
        from training import runners

        saved = {}

        class Model:
            def to(self, device):
                return self

            def parameters(self):
                return [object()]

            def load_state_dict(self, state):
                self.loaded_state = state

            def state_dict(self):
                return getattr(self, "loaded_state", {"final": True})

        class Torch:
            class optim:
                class Adam:
                    def __init__(self, parameters, lr):
                        pass

            @staticmethod
            def device(name):
                return name

            @staticmethod
            def save(payload, path):
                saved[path] = payload
                path.write_bytes(b"checkpoint")

        class Loader(list):
            dataset = [object()]

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            model = Model()
            config = SimpleNamespace(
                training=SimpleNamespace(lr=0.001, epochs=1, device="cpu", use_best_val_checkpoint=False),
                experiment=SimpleNamespace(output_dir=output_dir),
            )
            fake_module = types.SimpleNamespace(Stage1FlowClassifier=lambda: model)
            run = runners.TrainingRun(history=[{"epoch": 1, "val_accuracy": 0.9}], best_state={"best": True}, best_epoch=1, best_val_accuracy=0.9)

            with mock.patch.dict(sys.modules, {"models.stage1_r3d": fake_module}), mock.patch.object(runners, "_torch", return_value=Torch), mock.patch.object(runners, "_stage1_loader", return_value=Loader()), mock.patch.object(runners, "_classification_loop", return_value=run), mock.patch.object(runners, "_save_training_artifacts"):
                checkpoint = runners._train_stage1(config)

        self.assertEqual(saved[checkpoint]["model"], {"final": True})

    def test_epoch_summary_names_train_and_validation_metrics(self):
        from training.runners import _format_epoch_summary

        row = {
            "train_loss": 0.123456,
            "train_accuracy": 0.9,
            "val_loss": 0.234567,
            "val_accuracy": 0.8,
        }

        summary = _format_epoch_summary("stage1", epoch=2, epochs=80, row=row)

        self.assertEqual(
            summary,
            "stage1 epoch 2/80 train_loss=0.123456 train_acc=0.9000 val_loss=0.234567 val_acc=0.8000",
        )


if __name__ == "__main__":
    unittest.main()
