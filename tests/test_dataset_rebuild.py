import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


CANALS = ("LA", "LL", "LP", "RA", "RL", "RP")
POLARITIES = ("negative", "positive")


class DatasetRebuildTests(unittest.TestCase):
    def test_import_surface_uses_dataset_not_dataio(self):
        import importlib.util

        self.assertIsNotNone(importlib.util.find_spec("dataset"))
        self.assertIsNone(importlib.util.find_spec("dataio"))

    def test_rebuild_plan_deduplicates_copy_folders_without_touching_assets(self):
        from dataset.rebuild import build_rebuild_plan, execute_rebuild

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            _write_zip_fixture(root / "data" / "dataset.zip", include_copy_duplicate=True)

            plan = build_rebuild_plan(root, window_size=20, seed=42)
            result = execute_rebuild(plan, execute=False, generate=False)

            self.assertEqual(plan.window_size, 20)
            self.assertEqual(plan.total_sources, 37)
            self.assertEqual(plan.unique_samples, 36)
            self.assertEqual(plan.duplicate_samples, 1)
            self.assertEqual(result.moved_archives, 0)
            self.assertFalse((root / "data" / "_archive").exists())
            self.assertTrue((root / "data" / "dataset.zip").is_file())

    def test_rebuild_writes_active_raw_and_window20_splits_only(self):
        from dataset.rebuild import build_rebuild_plan, execute_rebuild

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            _write_zip_fixture(root / "data" / "dataset.zip")
            stale = root / "data" / "flow" / "old"
            stale.mkdir(parents=True)

            plan = build_rebuild_plan(root, window_size=20, seed=42)
            result = execute_rebuild(plan, execute=True, generate=False)

            splits = root / "data" / "label"
            raw_samples = root / "data" / "raw"

            self.assertEqual(result.unique_samples, 36)
            self.assertFalse((root / "data" / "_archive").exists())
            self.assertFalse((root / "metadata" / "dataset").exists())
            self.assertFalse(stale.exists())
            self.assertEqual(len(list(raw_samples.iterdir())), 36)

            train = set(np.load(splits / "train_names.npy", allow_pickle=True))
            val = set(np.load(splits / "val_names.npy", allow_pickle=True))
            test = set(np.load(splits / "test_names.npy", allow_pickle=True))
            self.assertEqual(len(train | val | test), 36)
            self.assertFalse(train & val)
            self.assertFalse(train & test)
            self.assertFalse(val & test)
            self.assertEqual(np.load(splits / "train_dir_onehot.npy").shape[1], 6)
            self.assertEqual(np.load(splits / "test_posneg_onehot_vor.npy").shape[1], 2)

    def test_artifact_image_size_uses_config_constant_without_data_image_size(self):
        from config import IMAGE_SIZE, load_config
        from dataset.rebuild import _artifact_image_size

        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.yaml"
            config_path.write_text("data:\n  root: data\n", encoding="utf-8")
            config = load_config(config_path, repo_root=Path(tmp))

            self.assertFalse(hasattr(config.data, "image_size"))
            self.assertEqual(_artifact_image_size(config), IMAGE_SIZE)


def _write_zip_fixture(path: Path, include_copy_duplicate: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    counter = 1
    with zipfile.ZipFile(path, "w") as zf:
        for canal in CANALS:
            for polarity in POLARITIES:
                for _ in range(3):
                    name = f"Z_{canal}_{polarity}__g{counter:03d}"
                    for frame in range(2):
                        zf.writestr(f"dataset_raw/{name}/{frame}.jpg", f"{name}-frame-{frame}".encode())
                    counter += 1
        if include_copy_duplicate:
            for frame in range(2):
                zf.writestr(
                    f"dataset_raw/Z_LA_negative__g001 copy/{frame}.jpg",
                    f"Z_LA_negative__g001-frame-{frame}".encode(),
                )


if __name__ == "__main__":
    unittest.main()
