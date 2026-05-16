from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path

from config import FLOW_FRAMES, IMAGE_SIZE
from dataset.datasets import DisplacementDataset, FlowDataset
from dataset.splits import load_split

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    def tqdm(iterable, **kwargs):
        return iterable


@dataclass(frozen=True)
class TrainingRun:
    history: list[dict[str, float]]
    best_state: dict | None = None
    best_epoch: int | None = None
    best_val_accuracy: float | None = None


def train(config, stage: str = "full") -> dict[str, Path]:
    outputs: dict[str, Path] = {}
    if stage in {"stage1", "full"}:
        outputs["stage1"] = _train_stage1(config)
    if stage in {"stage2", "full"}:
        stage1_checkpoint = _stage1_checkpoint_for_stage2(config, outputs)
        outputs["stage2"] = _train_stage2_bundle(config, stage1_checkpoint=stage1_checkpoint)
    return outputs


def _train_stage1(config) -> Path:
    torch = _torch()
    from models.stage1_r3d import Stage1FlowClassifier

    device = torch.device(_device_name(torch, config.training.device))
    train_loader = _stage1_loader(config, torch, subset="train", shuffle=True)
    val_loader = _stage1_loader(config, torch, subset="val", shuffle=False)
    model = Stage1FlowClassifier().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.lr)
    print(
        f"stage1 training: device={device} train_samples={len(train_loader.dataset)} "
        f"val_samples={len(val_loader.dataset)} train_batches={len(train_loader)} epochs={config.training.epochs}",
        flush=True,
    )
    use_best = _use_best_val_checkpoint(config)
    run = _classification_loop(
        model,
        train_loader,
        val_loader,
        optimizer,
        epochs=config.training.epochs,
        target="dir",
        device=device,
        name="stage1",
        use_best_val_checkpoint=use_best,
    )
    _save_training_artifacts(config.experiment.output_dir, "stage1", run.history)
    _apply_best_state(model, run, "stage1", use_best=use_best)
    output = config.experiment.output_dir / "checkpoints" / "stage1.pth"
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict()}, output)
    return output


def _stage1_checkpoint_for_stage2(config, outputs: dict[str, Path]) -> Path:
    if "stage1" in outputs:
        return outputs["stage1"]
    checkpoint = getattr(getattr(config, "checkpoints", None), "stage1", None)
    if checkpoint is not None and Path(checkpoint).is_file():
        return Path(checkpoint)
    print("stage2 evaluation requires stage1 SCC masks; training stage1 first.", flush=True)
    outputs["stage1"] = _train_stage1(config)
    return outputs["stage1"]


def _train_stage2_bundle(config, stage1_checkpoint: Path) -> Path:
    torch = _torch()
    from models.stage2_lstm import Stage2LSTM

    device = torch.device(_device_name(torch, config.training.device))
    model = Stage2LSTM().to(device)
    _train_stage2_phase(config, model, stage1_checkpoint, canal_group="vor", branches="double", name="stage2_vor", device=device)
    _train_stage2_phase(config, model, stage1_checkpoint, canal_group="lvl", branches="vertical", name="stage2_lvl", device=device)
    output = config.experiment.output_dir / "checkpoints" / "stage2.pth"
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "stage1_checkpoint": str(stage1_checkpoint)}, output)
    return output


def _train_stage2_phase(config, model, stage1_checkpoint: Path, canal_group: str, branches: str, name: str, device):
    torch = _torch()

    _set_stage2_trainable(model, branches)
    train_loader = _stage2_loader(config, torch, subset="train", canal_group=canal_group, shuffle=True, augment=True)
    val_loader = _stage2_loader(config, torch, subset="val", canal_group=canal_group, shuffle=False, augment=False)
    optimizer = torch.optim.Adam(_trainable_parameters(model), lr=config.training.lr)
    print(
        f"{name} training: device={device} train_samples={len(train_loader.dataset)} "
        f"val_samples={len(val_loader.dataset)} train_batches={len(train_loader)} epochs={config.training.epochs}",
        flush=True,
    )
    use_best = _use_best_val_checkpoint(config)
    run = _classification_loop(
        model,
        train_loader,
        val_loader,
        optimizer,
        epochs=config.training.epochs,
        target="posneg",
        device=device,
        name=name,
        direction_lookup=None,
        val_direction_lookup=None,
        use_best_val_checkpoint=use_best,
    )
    _save_training_artifacts(config.experiment.output_dir, name, run.history)
    _apply_best_state(model, run, name, use_best=use_best)


def _set_stage2_trainable(model, branches: str) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    names = {
        "double": ("double_lstm", "double_fc"),
        "vertical": ("left_lstm", "left_fc", "right_lstm", "right_fc"),
        "all": ("left_lstm", "left_fc", "right_lstm", "right_fc", "double_lstm", "double_fc"),
    }[branches]
    for name in names:
        for parameter in getattr(model, name).parameters():
            parameter.requires_grad = True


def _trainable_parameters(model):
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("No trainable Stage2 parameters selected.")
    return parameters


def _use_best_val_checkpoint(config) -> bool:
    return bool(getattr(getattr(config, "training", None), "use_best_val_checkpoint", False))


def _classification_loop(
    model,
    train_loader,
    val_loader,
    optimizer,
    epochs: int,
    target: str,
    device,
    name: str,
    direction_lookup=None,
    val_direction_lookup=None,
    use_best_val_checkpoint: bool = False,
) -> TrainingRun:
    torch = _torch()
    loss_fn = torch.nn.CrossEntropyLoss()
    model.train()
    history: list[dict[str, float]] = []
    best_state = None
    best_epoch = None
    best_val_accuracy = None
    total_batches = len(train_loader)
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        correct = 0
        total = 0
        progress = _progress_batches(enumerate(train_loader, start=1), total=total_batches, desc=f"{name} epoch {epoch}/{epochs}")
        for batch_index, batch in progress:
            optimizer.zero_grad()
            logits, labels = _forward_batch(model, batch, target, device, direction_lookup=direction_lookup)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            batch_size = int(labels.shape[0])
            total_loss += float(loss.item()) * batch_size
            correct += int((torch.argmax(logits, dim=1) == labels).sum().item())
            total += batch_size
            running_loss = total_loss / total if total else 0.0
            running_acc = correct / total if total else 0.0
            if hasattr(progress, "set_postfix"):
                progress.set_postfix(train_loss=f"{running_loss:.6f}", train_acc=f"{running_acc:.4f}")
        val_metrics = _evaluate_loader(model, val_loader, target, device, loss_fn, direction_lookup=val_direction_lookup)
        row = {
            "epoch": float(epoch),
            "train_loss": total_loss / total if total else 0.0,
            "train_accuracy": correct / total if total else 0.0,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
        }
        history.append(row)
        if use_best_val_checkpoint and (best_val_accuracy is None or row["val_accuracy"] > best_val_accuracy):
            best_val_accuracy = row["val_accuracy"]
            best_epoch = epoch
            best_state = _clone_state_dict(model)
        print(_format_epoch_summary(name, epoch, epochs, row), flush=True)
        model.train()
    return TrainingRun(history=history, best_state=best_state, best_epoch=best_epoch, best_val_accuracy=best_val_accuracy)


def _clone_state_dict(model) -> dict:
    cloned = {}
    for key, value in model.state_dict().items():
        if hasattr(value, "detach"):
            cloned[key] = value.detach().cpu().clone()
        else:
            cloned[key] = copy.deepcopy(value)
    return cloned


def _apply_best_state(model, run: TrainingRun, name: str, *, use_best: bool) -> None:
    if not use_best or run.best_state is None:
        return
    model.load_state_dict(run.best_state)
    print(
        f"{name} using best validation checkpoint: epoch={run.best_epoch} val_acc={run.best_val_accuracy:.4f}",
        flush=True,
    )


def _progress_batches(iterable, *, total: int, desc: str):
    return tqdm(iterable, total=total, desc=desc, leave=False, dynamic_ncols=True)


def _format_epoch_summary(name: str, epoch: int, epochs: int, row: dict[str, float]) -> str:
    return (
        f"{name} epoch {epoch}/{epochs} "
        f"train_loss={row['train_loss']:.6f} train_acc={row['train_accuracy']:.4f} "
        f"val_loss={row['val_loss']:.6f} val_acc={row['val_accuracy']:.4f}"
    )


def _evaluate_loader(model, loader, target: str, device, loss_fn, direction_lookup=None) -> dict[str, float]:
    torch = _torch()
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            logits, labels = _forward_batch(model, batch, target, device, direction_lookup=direction_lookup)
            loss = loss_fn(logits, labels)
            batch_size = int(labels.shape[0])
            total_loss += float(loss.item()) * batch_size
            correct += int((torch.argmax(logits, dim=1) == labels).sum().item())
            total += batch_size
    return {
        "loss": total_loss / total if total else 0.0,
        "accuracy": correct / total if total else 0.0,
    }


def _forward_batch(model, batch, target: str, device, direction_lookup=None):
    torch = _torch()
    if target == "dir":
        logits = model(batch.left_flow.to(device), batch.right_flow.to(device))
        labels = torch.argmax(batch.y_dir.to(device), dim=1)
    else:
        logits = model(_direction_input(batch, device, direction_lookup), batch.positions.to(device))
        labels = torch.argmax(batch.y_posneg.to(device), dim=1)
    return logits, labels


def _direction_input(batch, device, direction_lookup=None):
    if direction_lookup is None:
        return batch.y_dir.to(device)
    import torch

    directions = []
    for name in batch.name:
        key = str(name)
        if key not in direction_lookup:
            raise KeyError(f"Missing stage1 SCC mask for sample: {key}")
        directions.append(direction_lookup[key])
    return torch.stack(directions).to(device)


def _stage1_loader(config, torch, subset: str, shuffle: bool):
    split = load_split(config.data.label_dir, subset=subset, canal_group="all")
    dataset = FlowDataset(config.data.flow_dir(subset), split.names, split.dir_onehot, split.posneg_onehot, FLOW_FRAMES, IMAGE_SIZE)
    return torch.utils.data.DataLoader(dataset, batch_size=config.training.batch_size, shuffle=shuffle, num_workers=config.training.workers)


def _stage2_loader(config, torch, subset: str, canal_group: str, shuffle: bool, augment: bool):
    split = load_split(config.data.label_dir, subset=subset, canal_group=canal_group)
    dataset = DisplacementDataset(config.data.position_dir(subset), split.names, split.dir_onehot, split.posneg_onehot, augment=augment)
    return torch.utils.data.DataLoader(dataset, batch_size=config.training.batch_size, shuffle=shuffle, num_workers=config.training.workers)


def _save_training_artifacts(output_dir: Path, name: str, history: list[dict[str, float]]) -> None:
    metrics_dir = output_dir / "metrics"
    visuals_dir = output_dir / "visuals"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    visuals_dir.mkdir(parents=True, exist_ok=True)
    with open(metrics_dir / f"{name}_history.json", "w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
    _plot_training_curve(history, visuals_dir / f"{name}_training_curve.png", title=f"{name} training")


def _plot_training_curve(history: list[dict[str, float]], path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs, series = _accuracy_curve_series(history)
    colors = {"train accuracy": "#2563eb", "validation accuracy": "#dc2626"}
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, values in series:
        ax.plot(epochs, values, marker="o", linewidth=2.0, markersize=4.5, color=colors.get(label), label=label)
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, axis="y", color="#e5e7eb", linewidth=0.8)
    ax.legend(loc="best", frameon=False)
    ax.set_title(title.replace("training", "accuracy"))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _accuracy_curve_series(history: list[dict[str, float]]) -> tuple[list[int], list[tuple[str, list[float]]]]:
    epochs = [int(row["epoch"]) for row in history]
    series = [("train accuracy", [row.get("train_accuracy", row.get("accuracy", 0.0)) for row in history])]
    if any("val_accuracy" in row for row in history):
        series.append(("validation accuracy", [row.get("val_accuracy", 0.0) for row in history]))
    return epochs, series


def _torch():
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyTorch is required. Install the dependencies from requirements.txt, then run: python main.py") from exc

    return torch


def _device_name(torch, configured: str) -> str:
    configured = str(configured).lower()
    if configured == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return configured
