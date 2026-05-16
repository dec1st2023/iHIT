from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


WINDOW_SIZE = 20
FLOW_FRAMES = 15
IMAGE_SIZE = 224
RAFT_ROOT_NAME = "RAFT"


@dataclass(frozen=True)
class ExperimentConfig:
    output_dir: Path = Path("output")
    seed: int = 42


@dataclass(frozen=True)
class DataConfig:
    root: Path = Path("data")

    @property
    def raw_dir(self) -> Path:
        return self.root / "raw"

    @property
    def label_dir(self) -> Path:
        return self.root / "label"

    @property
    def flow_root(self) -> Path:
        return self.root / "flow"

    @property
    def position_root(self) -> Path:
        return self.root / "position"

    def flow_dir(self, split: str) -> Path:
        return self.flow_root / split

    def position_dir(self, split: str) -> Path:
        return self.position_root / split


@dataclass(frozen=True)
class CheckpointConfig:
    stage1: Path | None = None
    stage2: Path | None = None
    raft: Path | None = None


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 64
    workers: int = 0
    epochs: int = 80
    lr: float = 1e-3
    device: str = "auto"
    use_best_val_checkpoint: bool = False


@dataclass(frozen=True)
class IHITConfig:
    repo_root: Path
    mode: str = "all"
    stage: str = "full"
    train_stage: str = "full"
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    data: DataConfig = field(default_factory=DataConfig)
    checkpoints: CheckpointConfig = field(default_factory=CheckpointConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @property
    def raft_root(self) -> Path:
        return self.repo_root / RAFT_ROOT_NAME


def load_config(path: str | Path, repo_root: str | Path | None = None) -> IHITConfig:
    """Load a YAML experiment config and resolve relative paths."""

    try:
        import yaml
    except Exception as exc:  # pragma: no cover - dependency message only
        raise RuntimeError("PyYAML is required to load iHIT YAML configs.") from exc

    config_path = Path(path)
    if repo_root is None:
        root = Path.cwd()
    else:
        root = Path(repo_root)
    root = root.resolve()

    with open(config_path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    return IHITConfig(
        repo_root=root,
        mode=_choice(raw, "mode", "all", {"all", "train", "evaluate"}),
        stage=_choice(raw, "stage", "full", {"full", "stage1", "stage2"}),
        train_stage=_choice(raw, "train_stage", "full", {"full", "stage1", "stage2"}),
        experiment=_experiment(raw, root),
        data=_data(raw.get("data", {}), root),
        checkpoints=_checkpoints(raw.get("checkpoints", {}), root),
        training=_training(raw.get("training", {})),
    )


def _experiment(raw: dict[str, Any], root: Path) -> ExperimentConfig:
    return ExperimentConfig(
        output_dir=_resolve_path(raw.get("output_dir", "output"), root) or (root / "output"),
        seed=int(raw.get("seed", 42)),
    )


def _data(raw: dict[str, Any], root: Path) -> DataConfig:
    return DataConfig(
        root=_resolve_path(raw.get("root", "data"), root) or (root / "data"),
    )


def _checkpoints(raw: dict[str, Any], root: Path) -> CheckpointConfig:
    return CheckpointConfig(
        stage1=_resolve_path(raw.get("stage1"), root),
        stage2=_resolve_path(raw.get("stage2"), root),
        raft=_resolve_path(raw.get("raft"), root),
    )


def _training(raw: dict[str, Any]) -> TrainingConfig:
    return TrainingConfig(
        batch_size=int(raw.get("batch_size", 64)),
        workers=int(raw.get("workers", 0)),
        epochs=int(raw.get("epochs", 80)),
        lr=float(raw.get("lr", 1e-3)),
        device=str(raw.get("device", "auto")),
        use_best_val_checkpoint=_bool(raw.get("use_best_val_checkpoint", False)),
    )


def _choice(raw: dict[str, Any], key: str, default: str, allowed: set[str]) -> str:
    value = str(raw.get(key, default)).lower()
    if value not in allowed:
        choices = ", ".join(sorted(allowed))
        raise ValueError(f"{key} must be one of: {choices}")
    return value


def _resolve_path(value: Any, root: Path) -> Path | None:
    if value in (None, ""):
        return None
    raw = str(value).replace("\\", "/")
    path = Path(raw)
    if path.is_absolute():
        return path
    return (root / path).resolve()


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}
