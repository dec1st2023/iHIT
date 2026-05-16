from __future__ import annotations

from pathlib import Path

from config import WINDOW_SIZE
from dataset.rebuild import build_rebuild_plan, execute_rebuild


POSITION_FILES = ("leye_positions.pkl", "reye_positions.pkl", "nose_positions.pkl")


def ensure_data_ready(config, include_training: bool, include_evaluation: bool) -> bool:
    missing = _missing_assets(config, include_training=include_training, include_evaluation=include_evaluation)
    if not missing:
        print("Data assets: OK")
        return False

    print("Missing data assets:")
    for item in missing:
        print(f"- {item}")
    print("Rebuilding active dataset...")
    plan = build_rebuild_plan(config.repo_root, window_size=WINDOW_SIZE, seed=config.experiment.seed)
    execute_rebuild(plan, execute=True, generate=True, config=config)
    return True


def _missing_assets(config, include_training: bool, include_evaluation: bool) -> list[str]:
    missing: list[str] = []
    _require_nonempty_dir(missing, "raw samples", config.data.raw_dir)

    if include_training:
        _require_label_files(missing, config.data.label_dir, subset="train", stage=config.stage)
        _require_label_files(missing, config.data.label_dir, subset="val", stage=config.stage)
        if config.stage in {"stage1", "full"}:
            _require_nonempty_dir(missing, "training flow", config.data.flow_dir("train"))
            _require_nonempty_dir(missing, "validation flow", config.data.flow_dir("val"))
        if config.stage in {"stage2", "full"}:
            _require_position_files(missing, "training positions", config.data.position_dir("train"))
            _require_position_files(missing, "validation positions", config.data.position_dir("val"))

    if include_evaluation:
        _require_label_files(missing, config.data.label_dir, subset="test", stage=config.stage)
        if config.stage in {"stage1", "full"}:
            _require_nonempty_dir(missing, "test flow", config.data.flow_dir("test"))
        if config.stage in {"stage2", "full"}:
            _require_position_files(missing, "test positions", config.data.position_dir("test"))

    return missing


def _require_nonempty_dir(missing: list[str], label: str, path: Path | None) -> None:
    if path is None:
        missing.append(f"{label}: not configured")
    elif not path.is_dir():
        missing.append(f"{label}: {path}")
    elif not any(item.is_dir() for item in path.iterdir()):
        missing.append(f"{label} has no samples: {path}")


def _require_label_files(missing: list[str], path: Path | None, subset: str, stage: str) -> None:
    if path is None:
        missing.append("labels: not configured")
        return
    suffixes = []
    if stage in {"stage1", "full"}:
        suffixes.append("")
    if stage in {"stage2", "full"}:
        suffixes.extend(("_vor", "_lvl"))
    for suffix in suffixes:
        for label in ("names", "dir_onehot", "posneg_onehot"):
            filename = f"{subset}_{label}{suffix}.npy"
            if not (path / filename).is_file():
                missing.append(f"label file: {path / filename}")


def _require_position_files(missing: list[str], label: str, path: Path | None) -> None:
    if path is None:
        missing.append(f"{label}: not configured")
        return
    for filename in POSITION_FILES:
        if not (path / filename).is_file():
            missing.append(f"{label}: {path / filename}")
