from __future__ import annotations

import json
from dataclasses import dataclass
from dataclasses import asdict
from dataclasses import is_dataclass
from pathlib import Path

import numpy as np

from config import FLOW_FRAMES, IMAGE_SIZE
from dataset.datasets import DisplacementDataset, FlowDataset
from dataset.splits import load_split
from evaluation.metrics import classification_metrics
from progress import progress
from stage1_masks import stage1_direction_lookup


PAPER_CANAL_LABELS = ("LA", "LL", "LP", "RA", "RL", "RP")
PAPER_CONFUSION_LABELS = tuple(f"{canal}_{suffix}" for canal in PAPER_CANAL_LABELS for suffix in ("n", "p"))
STAGE1_CONFUSION_LABELS = ("LA", "LL", "LP", "RA", "RL", "RP")
POSNEG_CONFUSION_LABELS = ("negative", "positive")


@dataclass(frozen=True)
class Stage2Evaluation:
    metrics: object
    paper_confusion: np.ndarray


def evaluate(config, stage: str = "full", checkpoints: dict[str, Path] | None = None) -> Path:
    paper_confusion = None
    if stage == "stage1":
        metrics = _evaluate_stage1(config, _checkpoint(checkpoints, "stage1", config.checkpoints.stage1))
    elif stage == "stage2":
        stage1_path = _checkpoint(checkpoints, "stage1", config.checkpoints.stage1)
        stage2_path = _checkpoint(checkpoints, "stage2", config.checkpoints.stage2)
        vor = _evaluate_stage2(config, canal_group="vor", checkpoint_path=stage2_path, stage1_checkpoint_path=stage1_path)
        lvl = _evaluate_stage2(config, canal_group="lvl", checkpoint_path=stage2_path, stage1_checkpoint_path=stage1_path)
        metrics = {
            "stage2_vor": asdict(vor.metrics),
            "stage2_lvl": asdict(lvl.metrics),
        }
        paper_confusion = vor.paper_confusion + lvl.paper_confusion
    else:
        stage1_path = _checkpoint(checkpoints, "stage1", config.checkpoints.stage1)
        stage2_path = _checkpoint(checkpoints, "stage2", config.checkpoints.stage2)
        vor = _evaluate_stage2(config, canal_group="vor", checkpoint_path=stage2_path, stage1_checkpoint_path=stage1_path)
        lvl = _evaluate_stage2(config, canal_group="lvl", checkpoint_path=stage2_path, stage1_checkpoint_path=stage1_path)
        metrics = {
            "stage1": asdict(_evaluate_stage1(config, stage1_path)),
            "stage2_vor": asdict(vor.metrics),
            "stage2_lvl": asdict(lvl.metrics),
        }
        paper_confusion = vor.paper_confusion + lvl.paper_confusion
    output_dir = config.experiment.output_dir / "metrics"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{stage}_metrics.json"
    _write_json(output_path, metrics)
    _save_evaluation_visuals(config.experiment.output_dir, metrics, prefix=stage, paper_confusion=paper_confusion)
    return output_path


def _evaluate_stage1(config, checkpoint_path):
    torch = _torch()
    from models.stage1_r3d import Stage1FlowClassifier

    split = load_split(config.data.label_dir, subset="test", canal_group="all")
    dataset = FlowDataset(config.data.flow_dir("test"), split.names, split.dir_onehot, split.posneg_onehot, FLOW_FRAMES, IMAGE_SIZE)
    loader = torch.utils.data.DataLoader(dataset, batch_size=config.training.batch_size, shuffle=False, num_workers=config.training.workers)
    model = Stage1FlowClassifier()
    _load_checkpoint(model, checkpoint_path)
    model.eval()
    true, pred = [], []
    with torch.no_grad():
        for batch in _progress_batches(loader, desc="stage1 test evaluation", total=len(loader)):
            logits = model(batch.left_flow, batch.right_flow)
            true.extend(torch.argmax(batch.y_dir, dim=1).cpu().numpy().tolist())
            pred.extend(torch.argmax(logits, dim=1).cpu().numpy().tolist())
    return classification_metrics(true, pred, num_classes=6)


def _evaluate_stage2(config, canal_group: str = "all", checkpoint_path=None, stage1_checkpoint_path=None):
    torch = _torch()
    from models.stage2_lstm import Stage2LSTM

    split = load_split(config.data.label_dir, subset="test", canal_group=canal_group)
    dataset = DisplacementDataset(config.data.position_dir("test"), split.names, split.dir_onehot, split.posneg_onehot, augment=False)
    loader = torch.utils.data.DataLoader(dataset, batch_size=config.training.batch_size, shuffle=False, num_workers=config.training.workers)
    model = Stage2LSTM()
    _load_checkpoint(model, checkpoint_path)
    model.eval()
    direction_lookup = stage1_direction_lookup(config, torch, stage1_checkpoint_path, subset="test", canal_group=canal_group, device="cpu")
    true, pred, true_canals = [], [], []
    with torch.no_grad():
        for batch in _progress_batches(loader, desc=f"stage2_{canal_group} test evaluation", total=len(loader)):
            logits = model(_direction_input(torch, batch, direction_lookup), batch.positions)
            true.extend(torch.argmax(batch.y_posneg, dim=1).cpu().numpy().tolist())
            pred.extend(torch.argmax(logits, dim=1).cpu().numpy().tolist())
            true_canals.extend(torch.argmax(batch.y_dir, dim=1).cpu().numpy().tolist())
    return Stage2Evaluation(
        classification_metrics(true, pred, num_classes=2),
        _stage2_confusion_matrix(true_canals, true, pred),
    )


def _load_checkpoint(model, path):
    if path is None:
        raise ValueError("Evaluation checkpoint is not configured.")
    torch = _torch()
    checkpoint = _load_weights(torch, path)
    state = _checkpoint_state(checkpoint)
    model.load_state_dict(state, strict=False)


def _checkpoint_state(checkpoint):
    if isinstance(checkpoint, dict):
        return checkpoint.get("model", checkpoint)
    return checkpoint


def _direction_input(torch, batch, direction_lookup):
    directions = []
    for name in batch.name:
        key = str(name)
        if key not in direction_lookup:
            raise KeyError(f"Missing stage1 SCC mask for sample: {key}")
        directions.append(direction_lookup[key])
    return torch.stack(directions)


def _progress_batches(iterable, *, desc: str, total: int):
    return progress(iterable, desc=desc, unit="batch", total=total, leave=False, dynamic_ncols=True)


def _load_weights(torch, path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _checkpoint(checkpoints: dict[str, Path] | None, key: str, default: Path | None) -> Path | None:
    if checkpoints and key in checkpoints:
        return checkpoints[key]
    return default


def _write_json(path: Path, payload) -> None:
    def default(value):
        if is_dataclass(value):
            return asdict(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        return value

    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=default)


def _save_evaluation_visuals(output_dir: Path, metrics, prefix: str = "stage", paper_confusion: np.ndarray | None = None) -> None:
    _save_confusion_visuals(output_dir, metrics, prefix=prefix)
    if paper_confusion is not None:
        _save_paper_confusion_visual(output_dir, paper_confusion)
    _save_metric_summary_visual(output_dir, metrics)


def _save_confusion_visuals(output_dir: Path, metrics, prefix: str = "stage") -> None:
    visuals_dir = output_dir / "visuals"
    visuals_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(metrics, "confusion"):
        _plot_confusion(metrics.confusion, visuals_dir / _confusion_filename(prefix), f"{prefix} confusion matrix")
        return
    for name, item in metrics.items():
        if hasattr(item, "confusion"):
            _plot_confusion(item.confusion, visuals_dir / _confusion_filename(name), f"{name} confusion matrix")
        elif isinstance(item, dict) and "confusion" in item:
            _plot_confusion(item["confusion"], visuals_dir / _confusion_filename(name), f"{name} confusion matrix")


def _confusion_filename(name: str) -> str:
    if name == "stage2":
        return "stage2_posneg_conf_mat.png"
    return f"{name}_conf_mat.png"


def _plot_confusion(confusion, path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    matrix = np.asarray(confusion, dtype=np.float64)
    counts = np.asarray(confusion, dtype=np.int64)
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums != 0)
    labels = _confusion_labels(matrix.shape[0])
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    image = ax.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = normalized[row, col]
            color = "white" if value >= 0.55 else "#111827"
            ax.text(col, row, f"{value:.2f}\n({counts[row, col]})", ha="center", va="center", color=color, fontsize=8)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("row-normalized")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _confusion_labels(size: int) -> tuple[str, ...]:
    if size == 6:
        return STAGE1_CONFUSION_LABELS
    if size == 2:
        return POSNEG_CONFUSION_LABELS
    return tuple(str(index) for index in range(size))


def _save_metric_summary_visual(output_dir: Path, metrics) -> None:
    rows = _metric_summary_rows(metrics)
    if not rows:
        return
    visuals_dir = output_dir / "visuals"
    visuals_dir.mkdir(parents=True, exist_ok=True)
    _plot_metric_summary(rows, visuals_dir / "evaluation_metrics.png")


def _metric_summary_rows(metrics) -> list[dict[str, float | str]]:
    if _has_metric_fields(metrics):
        return [_metric_row("evaluation", metrics)]
    rows = []
    if isinstance(metrics, dict):
        for name, item in metrics.items():
            if _has_metric_fields(item):
                rows.append(_metric_row(str(name), item))
    return rows


def _has_metric_fields(item) -> bool:
    mapping = _metric_mapping(item)
    return all(mapping.get(key) is not None for key in ("accuracy", "precision_macro", "recall_macro", "f1_macro"))


def _metric_row(name: str, item) -> dict[str, float | str]:
    mapping = _metric_mapping(item)
    return {
        "name": name,
        "accuracy": float(mapping["accuracy"]),
        "precision": float(mapping["precision_macro"]),
        "recall": float(mapping["recall_macro"]),
        "f1": float(mapping["f1_macro"]),
    }


def _metric_mapping(item) -> dict:
    if is_dataclass(item):
        return asdict(item)
    if isinstance(item, dict):
        return item
    return {
        "accuracy": getattr(item, "accuracy", None),
        "precision_macro": getattr(item, "precision_macro", None),
        "recall_macro": getattr(item, "recall_macro", None),
        "f1_macro": getattr(item, "f1_macro", None),
    }


def _plot_metric_summary(rows: list[dict[str, float | str]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    metric_names = ("accuracy", "precision", "recall", "f1")
    labels = [str(row["name"]) for row in rows]
    values = np.asarray([[float(row[metric]) for metric in metric_names] for row in rows], dtype=np.float64)
    x = np.arange(len(labels))
    width = min(0.18, 0.75 / len(metric_names))
    colors = ("#2563eb", "#0891b2", "#16a34a", "#dc2626")
    fig_width = max(7.0, 1.6 * len(labels) + 3.5)
    fig, ax = plt.subplots(figsize=(fig_width, 4.6))
    for index, metric in enumerate(metric_names):
        offset = (index - (len(metric_names) - 1) / 2) * width
        bars = ax.bar(x + offset, values[:, index], width=width, label=metric, color=colors[index], edgecolor="white", linewidth=0.8)
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, min(0.98, height + 0.025), f"{height:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("score")
    ax.set_title("Evaluation Metrics")
    ax.grid(True, axis="y", color="#e5e7eb", linewidth=0.8)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=len(metric_names), frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _stage2_confusion_matrix(true_canals, true_posneg, pred_posneg) -> np.ndarray:
    matrix = np.zeros((12, 12), dtype=np.int64)
    for canal, actual_posneg, predicted_posneg in zip(true_canals, true_posneg, pred_posneg):
        actual = int(canal) * 2 + int(actual_posneg)
        predicted = int(canal) * 2 + int(predicted_posneg)
        matrix[actual, predicted] += 1
    return matrix


def _save_paper_confusion_visual(output_dir: Path, confusion: np.ndarray) -> None:
    visuals_dir = output_dir / "visuals"
    visuals_dir.mkdir(parents=True, exist_ok=True)
    _plot_paper_confusion(confusion, visuals_dir / "stage2_conf_mat.png")


def _plot_paper_confusion(confusion: np.ndarray, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matrix = np.asarray(confusion, dtype=np.float64)
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums != 0)
    fig, ax = plt.subplots(figsize=(8.2, 7.2))
    image = ax.imshow(normalized, cmap="Oranges", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(PAPER_CONFUSION_LABELS)))
    ax.set_yticks(np.arange(len(PAPER_CONFUSION_LABELS)))
    ax.set_xticklabels(PAPER_CONFUSION_LABELS, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(PAPER_CONFUSION_LABELS, rotation=45, va="center", fontsize=8)
    ax.set_xlabel("Predicted Labels", fontsize=12)
    ax.set_ylabel("True Labels", fontsize=12)
    for row in range(normalized.shape[0]):
        for col in range(normalized.shape[1]):
            ax.text(col, row, f"{normalized[row, col]:.2f}", ha="center", va="center", fontsize=6, color="#222222")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _torch():
    import torch

    return torch
