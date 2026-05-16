from __future__ import annotations

import hashlib
import re
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from config import IMAGE_SIZE, WINDOW_SIZE
from progress import progress


CANALS = ("LA", "LL", "LP", "RA", "RL", "RP")
POLARITIES = ("negative", "positive")
VOR_CANALS = {"LL", "RL"}
LVL_CANALS = {"LA", "LP", "RA", "RP"}
SAMPLE_RE = re.compile(r"^Z_(LA|LL|LP|RA|RL|RP)_(negative|positive)__g(\d+)(?: copy)?$")


@dataclass(frozen=True)
class SampleSource:
    source_id: str
    original_name: str
    sample_id: str
    canal: str
    posneg: str
    digest: str
    kind: str
    source_path: Path
    zip_members: tuple[str, ...] = ()
    is_copy: bool = False


@dataclass(frozen=True)
class DuplicateRecord:
    duplicate_source_id: str
    duplicate_original_name: str
    duplicate_source_path: Path
    kept_source_id: str
    kept_sample_id: str
    kept_original_name: str
    digest: str


@dataclass(frozen=True)
class RebuildPlan:
    repo_root: Path
    window_size: int
    seed: int
    dataset_root: Path
    build_root: Path
    zip_source: Path | None
    unique: tuple[SampleSource, ...]
    duplicates: tuple[DuplicateRecord, ...]
    splits: dict[str, tuple[str, ...]]

    @property
    def total_sources(self) -> int:
        return len(self.unique) + len(self.duplicates)

    @property
    def unique_samples(self) -> int:
        return len(self.unique)

    @property
    def duplicate_samples(self) -> int:
        return len(self.duplicates)


@dataclass(frozen=True)
class RebuildResult:
    executed: bool
    unique_samples: int
    duplicate_samples: int
    moved_archives: int
    generated: bool
    messages: tuple[str, ...]


def build_rebuild_plan(repo_root: str | Path, window_size: int = WINDOW_SIZE, seed: int = 42) -> RebuildPlan:
    root = Path(repo_root).resolve()
    dataset_root = root / "data"
    build_root = root / ".cache" / "dataset_build" / f"w{window_size}"
    zip_source = _first_existing(root / "data" / "dataset.zip")

    candidates = list(_iter_zip_sources(zip_source)) if zip_source else []
    if not candidates:
        candidates.extend(_iter_directory_sources(dataset_root / "raw"))

    unique, duplicates = _deduplicate_sources(candidates)
    splits = _make_splits(unique, seed)
    return RebuildPlan(
        repo_root=root,
        window_size=window_size,
        seed=seed,
        dataset_root=dataset_root,
        build_root=build_root,
        zip_source=zip_source,
        unique=tuple(unique),
        duplicates=tuple(duplicates),
        splits=splits,
    )


def execute_rebuild(plan: RebuildPlan, execute: bool = False, generate: bool = False, config=None) -> RebuildResult:
    messages = [_format_plan_head(plan)]
    if generate and config is None:
        raise ValueError("config is required when generate=True")
    if execute and not plan.unique:
        raise FileNotFoundError("No raw samples found. Add data/dataset.zip or data/raw before rebuilding.")
    if execute and generate:
        _preflight_generation(config)
    if not execute:
        return RebuildResult(False, plan.unique_samples, plan.duplicate_samples, 0, False, tuple(messages))

    _assert_inside(plan.repo_root, plan.build_root)
    if plan.build_root.exists():
        shutil.rmtree(plan.build_root)
    raw_samples = plan.build_root / "raw"
    raw_samples.mkdir(parents=True, exist_ok=True)

    for sample in progress(plan.unique, desc="stage raw samples", unit="sample", total=len(plan.unique)):
        _copy_sample(sample, raw_samples / sample.sample_id)
    print(f"Staged {plan.unique_samples} raw samples into {raw_samples}", flush=True)
    if generate:
        _generate_artifacts(plan, config, raw_samples)

    moved_archives = _replace_active_dataset(plan)
    _write_splits(plan)
    messages.append(f"active dataset: {plan.dataset_root}")
    return RebuildResult(True, plan.unique_samples, plan.duplicate_samples, moved_archives, generate, tuple(messages))


def format_rebuild_result(result: RebuildResult) -> str:
    lines = list(result.messages)
    lines.extend(
        [
            f"Unique samples: {result.unique_samples}",
            f"Duplicate samples: {result.duplicate_samples}",
            f"Backups created: {result.moved_archives}",
            f"Generated artifacts: {'yes' if result.generated else 'no'}",
        ]
    )
    return "\n".join(lines)


def _iter_zip_sources(zip_path: Path) -> Iterable[SampleSource]:
    grouped: dict[str, list[str]] = {}
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            parts = name.split("/")
            if len(parts) < 3 or parts[0] != "dataset_raw" or name.endswith("/"):
                continue
            grouped.setdefault(parts[1], []).append(name)
        for sample_name, members in sorted(grouped.items()):
            parsed = _parse_sample_name(sample_name)
            if parsed is None:
                continue
            digest = _hash_zip_members(zf, members)
            yield SampleSource(
                source_id=f"zip:{sample_name}",
                original_name=sample_name,
                sample_id=_sample_id(sample_name),
                canal=parsed[0],
                posneg=parsed[1],
                digest=digest,
                kind="zip",
                source_path=zip_path,
                zip_members=tuple(sorted(members)),
                is_copy=" copy" in sample_name,
            )


def _iter_directory_sources(root: Path) -> Iterable[SampleSource]:
    if not root.is_dir():
        return
    for folder in sorted(path for path in root.iterdir() if path.is_dir()):
        parsed = _parse_sample_name(folder.name)
        if parsed is None:
            continue
        yield SampleSource(
            source_id=f"dir:{root.name}:{folder.name}",
            original_name=folder.name,
            sample_id=_sample_id(folder.name),
            canal=parsed[0],
            posneg=parsed[1],
            digest=_hash_directory(folder),
            kind="directory",
            source_path=folder,
            is_copy=" copy" in folder.name,
        )


def _deduplicate_sources(candidates: list[SampleSource]) -> tuple[list[SampleSource], list[DuplicateRecord]]:
    by_digest: dict[str, list[SampleSource]] = {}
    for source in candidates:
        by_digest.setdefault(source.digest, []).append(source)
    unique: list[SampleSource] = []
    duplicates: list[DuplicateRecord] = []
    used_ids: set[str] = set()
    for digest, group in sorted(by_digest.items(), key=lambda item: item[1][0].original_name):
        ordered = sorted(group, key=lambda item: (item.is_copy, item.kind != "zip", item.original_name))
        kept = _with_unique_sample_id(ordered[0], used_ids)
        unique.append(kept)
        for duplicate in ordered[1:]:
            duplicates.append(
                DuplicateRecord(
                    duplicate_source_id=duplicate.source_id,
                    duplicate_original_name=duplicate.original_name,
                    duplicate_source_path=duplicate.source_path,
                    kept_source_id=kept.source_id,
                    kept_sample_id=kept.sample_id,
                    kept_original_name=kept.original_name,
                    digest=digest,
                )
            )
    return sorted(unique, key=lambda item: item.sample_id), duplicates


def _make_splits(samples: Iterable[SampleSource], seed: int) -> dict[str, tuple[str, ...]]:
    rng = np.random.default_rng(seed)
    grouped: dict[tuple[str, str], list[str]] = {}
    for sample in samples:
        grouped.setdefault((sample.canal, sample.posneg), []).append(sample.sample_id)
    splits = {"train": [], "val": [], "test": []}
    for key in sorted(grouped):
        names = np.array(sorted(grouped[key]), dtype=object)
        rng.shuffle(names)
        n = len(names)
        if n >= 3:
            val_count = max(1, int(round(n * 0.1)))
            test_count = max(1, int(round(n * 0.1)))
            if val_count + test_count >= n:
                val_count = test_count = 1
        elif n == 2:
            val_count, test_count = 1, 0
        else:
            val_count = test_count = 0
        train_count = n - val_count - test_count
        splits["train"].extend(names[:train_count].tolist())
        splits["val"].extend(names[train_count : train_count + val_count].tolist())
        splits["test"].extend(names[train_count + val_count :].tolist())
    return {name: tuple(sorted(values)) for name, values in splits.items()}


def _write_splits(plan: RebuildPlan) -> None:
    splits_dir = plan.dataset_root / "label"
    if splits_dir.exists():
        shutil.rmtree(splits_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)
    by_id = {sample.sample_id: sample for sample in plan.unique}
    for split, names in plan.splits.items():
        _write_split_arrays(splits_dir, split, names, by_id, canal_filter=None, suffix="")
        _write_split_arrays(splits_dir, split, names, by_id, canal_filter=VOR_CANALS, suffix="_vor")
        _write_split_arrays(splits_dir, split, names, by_id, canal_filter=LVL_CANALS, suffix="_lvl")


def _write_split_arrays(
    split_dir: Path,
    split: str,
    names: Iterable[str],
    by_id: dict[str, SampleSource],
    canal_filter: set[str] | None,
    suffix: str,
) -> None:
    selected = [name for name in names if canal_filter is None or by_id[name].canal in canal_filter]
    np.save(split_dir / f"{split}_names{suffix}.npy", np.array(selected, dtype=object))
    np.save(split_dir / f"{split}_dir_onehot{suffix}.npy", np.array([_onehot(CANALS, by_id[name].canal) for name in selected], dtype=np.float32))
    np.save(
        split_dir / f"{split}_posneg_onehot{suffix}.npy",
        np.array([_onehot(POLARITIES, by_id[name].posneg) for name in selected], dtype=np.float32),
    )


def _copy_sample(sample: SampleSource, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    if sample.kind == "zip":
        with zipfile.ZipFile(sample.source_path) as zf:
            prefix = f"dataset_raw/{sample.original_name}/"
            for member in sample.zip_members:
                rel = member.removeprefix(prefix)
                if not rel:
                    continue
                target = destination / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)
    else:
        shutil.copytree(sample.source_path, destination, dirs_exist_ok=True)


def _generate_artifacts(plan: RebuildPlan, config, raw_samples: Path) -> None:
    from preprocess.flow import generate_flow_dataset
    from preprocess.landmarks import process_eyes, process_nose
    from flow.raft_adapter import RAFTAdapter

    adapter = RAFTAdapter(config.raft_root, config.checkpoints.raft, device=config.training.device)
    model = adapter.load_model()
    with adapter.core_imports():
        from raft_utils import flow_viz
        from raft_utils.utils import InputPadder

    split_items = list(plan.splits.items())
    for split, names in progress(split_items, desc="artifact splits", unit="split", total=len(split_items)):
        print(f"Generating window-{plan.window_size} artifacts for {split}: {len(names)} samples", flush=True)
        position_dir = plan.build_root / "position" / split
        flow_dir = plan.build_root / "flow" / split
        print(f"  landmarks: {position_dir}", flush=True)
        process_eyes(raw_samples, position_dir, window_size=plan.window_size, sample_names=names)
        process_nose(raw_samples, position_dir, sample_names=names)
        print(f"  flow: {flow_dir}", flush=True)
        generate_flow_dataset(
            raw_samples,
            flow_dir,
            model,
            flow_viz,
            InputPadder,
            window_size=plan.window_size,
            sample_names=names,
            image_size=_artifact_image_size(config),
        )


def _artifact_image_size(config) -> int:
    return int(getattr(config.data, "image_size", IMAGE_SIZE))


def _replace_active_dataset(plan: RebuildPlan) -> int:
    plan.dataset_root.mkdir(parents=True, exist_ok=True)
    for name in ("raw", "position", "positions", "flow", "label"):
        child = plan.dataset_root / name
        if not child.exists():
            continue
        _assert_inside(plan.repo_root, child)
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
    for child in plan.build_root.iterdir():
        destination = plan.dataset_root / child.name
        if destination.exists():
            if destination.is_dir():
                shutil.rmtree(destination)
            else:
                destination.unlink()
        shutil.move(str(child), str(destination))
    shutil.rmtree(plan.build_root, ignore_errors=True)
    return 0


def _preflight_generation(config) -> None:
    missing = []
    for module in ("cv2", "mediapipe", "torch"):
        try:
            __import__(module)
        except Exception:
            missing.append(module)
    try:
        from modelscope.pipelines import pipeline as _pipeline  # noqa: F401
        from modelscope.utils.constant import Tasks as _tasks  # noqa: F401
    except Exception as exc:
        missing.append(f"modelscope.pipelines ({exc})")
    try:
        from preprocess.modelscope_scrfd_windows import check_runtime

        check_runtime()
    except Exception as exc:
        missing.append(f"Windows ModelScope SCRFD ({exc})")
    if missing:
        raise RuntimeError(f"Missing generation dependencies: {', '.join(missing)}")
    if config.checkpoints.raft is None or not config.checkpoints.raft.is_file():
        raise FileNotFoundError(f"RAFT checkpoint not found: {config.checkpoints.raft}")
    if not config.raft_root.is_dir():
        raise FileNotFoundError(f"RAFT root not found: {config.raft_root}")


def _hash_zip_members(zf: zipfile.ZipFile, members: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for member in sorted(members):
        rel = "/".join(member.split("/")[2:])
        digest.update(rel.encode("utf-8"))
        with zf.open(member) as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _hash_directory(path: Path) -> str:
    digest = hashlib.sha256()
    for file in sorted(item for item in path.rglob("*") if item.is_file()):
        rel = file.relative_to(path).as_posix()
        digest.update(rel.encode("utf-8"))
        with open(file, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _parse_sample_name(name: str) -> tuple[str, str] | None:
    match = SAMPLE_RE.match(name)
    if not match:
        return None
    return match.group(1), match.group(2)


def _sample_id(original_name: str) -> str:
    return original_name.replace(" copy", "_copy").replace(" ", "_")


def _with_unique_sample_id(source: SampleSource, used_ids: set[str]) -> SampleSource:
    base = source.sample_id
    sample_id = base
    index = 2
    while sample_id in used_ids:
        sample_id = f"{base}_{index}"
        index += 1
    used_ids.add(sample_id)
    if sample_id == source.sample_id:
        return source
    return SampleSource(
        source.source_id,
        source.original_name,
        sample_id,
        source.canal,
        source.posneg,
        source.digest,
        source.kind,
        source.source_path,
        source.zip_members,
        source.is_copy,
    )


def _onehot(labels: tuple[str, ...], value: str) -> list[float]:
    return [1.0 if item == value else 0.0 for item in labels]


def _format_plan_head(plan: RebuildPlan) -> str:
    return (
        f"Rebuild plan: window={plan.window_size}, total_sources={plan.total_sources}, "
        f"unique={plan.unique_samples}, duplicates={plan.duplicate_samples}"
    )


def _first_existing(*paths: Path) -> Path | None:
    for path in paths:
        if path.is_file():
            return path
    return None

def _assert_inside(root: Path, path: Path) -> None:
    resolved_root = root.resolve()
    resolved_path = path.resolve()
    if resolved_path != resolved_root and resolved_root not in resolved_path.parents:
        raise ValueError(f"Refusing to operate outside repository: {resolved_path}")
