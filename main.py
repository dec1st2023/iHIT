from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config import load_config  # noqa: E402
import workflow  # noqa: E402


def main() -> int:
    _require_runtime_dependencies()
    config = load_config(ROOT / "config.yaml", repo_root=ROOT)
    result = workflow.run(config)
    if result.metrics_path is not None:
        print(f"Metrics: {result.metrics_path}")
    return 0


def _require_runtime_dependencies() -> None:
    missing = [name for name in ("torch",) if importlib.util.find_spec(name) is None]
    if not missing:
        return
    packages = ", ".join(missing)
    raise RuntimeError(
        f"Missing required runtime package(s): {packages}. "
        "Install the dependencies from requirements.txt, then run: python main.py"
    )



if __name__ == "__main__":
    raise SystemExit(main())
