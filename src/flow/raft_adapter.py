from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path


class RAFTAdapter:
    """Encapsulated access to the vendored/local RAFT implementation."""

    def __init__(self, raft_root: str | Path, checkpoint: str | Path, device: str = "auto"):
        self.raft_root = Path(raft_root)
        self.checkpoint = Path(checkpoint)
        self.device = device

    def validate(self) -> None:
        core = self.raft_root / "core"
        if not core.is_dir():
            raise FileNotFoundError(f"RAFT core directory not found: {core}")
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f"RAFT checkpoint not found: {self.checkpoint}")

    def load_model(self):
        self.validate()
        with _temporary_sys_path(self.raft_root / "core"):
            import torch
            from raft import RAFT

            args = _RAFTArgs()
            model = RAFT(args)
            device = _device_name(torch, self.device)
            state = _strip_module_prefix(torch.load(self.checkpoint, map_location=device))
            model.load_state_dict(state)
            model.to(device)
            model.eval()
            return model

    @contextmanager
    def core_imports(self):
        self.validate()
        with _temporary_sys_path(self.raft_root / "core"):
            yield


class _RAFTArgs:
    small = False
    mixed_precision = False
    alternate_corr = False

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)


def _strip_module_prefix(state):
    if not isinstance(state, dict) or not state:
        return state
    if not all(isinstance(key, str) and key.startswith("module.") for key in state):
        return state
    return {key.removeprefix("module."): value for key, value in state.items()}


def _device_name(torch, configured: str) -> str:
    configured = str(configured).lower()
    if configured == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return configured


@contextmanager
def _temporary_sys_path(path: Path):
    text = str(path)
    inserted = text not in sys.path
    if inserted:
        sys.path.insert(0, text)
    try:
        yield
    finally:
        if inserted:
            sys.path.remove(text)
