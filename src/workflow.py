from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path

from config import ExperimentConfig
from data_setup import ensure_data_ready
from evaluation.runners import evaluate
from training.runners import train


@dataclass(frozen=True)
class WorkflowResult:
    trained_checkpoints: dict[str, Path] = field(default_factory=dict)
    metrics_path: Path | None = None
    output_dir: Path | None = None


def run(config) -> WorkflowResult:
    config = _with_case_output_dir(config)
    print(f"Output directory: {config.experiment.output_dir}", flush=True)

    if config.mode == "train":
        train_config = _with_training_stage(config)
        ensure_data_ready(train_config, include_training=True, include_evaluation=False)
        checkpoints = train(train_config, stage=train_config.stage)
        return WorkflowResult(trained_checkpoints=checkpoints, output_dir=train_config.experiment.output_dir)

    if config.mode == "evaluate":
        ensure_data_ready(config, include_training=False, include_evaluation=True)
        _ensure_configured_checkpoints(config)
        metrics_path = evaluate(config, stage=config.stage, checkpoints=None)
        return WorkflowResult(metrics_path=metrics_path, output_dir=config.experiment.output_dir)

    if config.mode == "all":
        train_config = _with_training_stage(config)
        ensure_data_ready(train_config, include_training=True, include_evaluation=True)
        checkpoints = train(train_config, stage=train_config.stage)
        metrics_path = evaluate(train_config, stage=train_config.stage, checkpoints=checkpoints)
        return WorkflowResult(trained_checkpoints=checkpoints, metrics_path=metrics_path, output_dir=train_config.experiment.output_dir)

    raise ValueError(f"Unknown mode: {config.mode}")


def _ensure_configured_checkpoints(config) -> None:
    if config.stage in {"stage1", "stage2", "full"}:
        _require_file("stage1 checkpoint", config.checkpoints.stage1)
    if config.stage in {"stage2", "full"}:
        _require_file("stage2 checkpoint", config.checkpoints.stage2)


def _require_file(label: str, path: Path | None) -> None:
    if path is None or not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")


def _with_case_output_dir(config):
    case_dir = config.experiment.output_dir / f"case_{_timestamp()}"
    case_dir.mkdir(parents=True, exist_ok=True)
    experiment = ExperimentConfig(output_dir=case_dir, seed=config.experiment.seed)
    return replace(config, experiment=experiment)


def _with_training_stage(config):
    return replace(config, stage=config.train_stage)


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")
