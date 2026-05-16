from __future__ import annotations

from pathlib import Path

from config import FLOW_FRAMES, IMAGE_SIZE
from dataset.datasets import FlowDataset
from dataset.splits import load_split
from progress import progress


def stage1_direction_lookup(config, torch, checkpoint_path: Path | None, subset: str, canal_group: str, device) -> dict[str, object]:
    if checkpoint_path is None:
        raise ValueError("Stage2 requires a stage1 checkpoint for SCC mask generation.")

    from models.stage1_r3d import Stage1FlowClassifier

    split = load_split(config.data.label_dir, subset=subset, canal_group=canal_group)
    dataset = FlowDataset(config.data.flow_dir(subset), split.names, split.dir_onehot, split.posneg_onehot, FLOW_FRAMES, IMAGE_SIZE)
    loader = torch.utils.data.DataLoader(dataset, batch_size=config.training.batch_size, shuffle=False, num_workers=config.training.workers)
    model = Stage1FlowClassifier()
    _load_checkpoint(torch, model, checkpoint_path)
    model.to(device)
    model.eval()

    lookup = {}
    with torch.no_grad():
        for batch in _progress_batches(loader, desc=f"stage1 mask {subset} {canal_group}", total=len(loader)):
            logits = model(batch.left_flow.to(device), batch.right_flow.to(device))
            predictions = torch.argmax(logits, dim=1)
            directions = torch.nn.functional.one_hot(predictions, num_classes=6).float().cpu()
            for name, direction in zip(batch.name, directions):
                lookup[str(name)] = direction
    return lookup


def _load_checkpoint(torch, model, path: Path) -> None:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    state = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state, strict=False)


def _progress_batches(iterable, *, desc: str, total: int):
    return progress(iterable, desc=desc, unit="batch", total=total, leave=False, dynamic_ncols=True)
