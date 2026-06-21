from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
STATION_PY_ROOT = REPO_ROOT / "software" / "station" / "shared"


def setup_import_paths() -> None:
    for path in (REPO_ROOT, STATION_PY_ROOT):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
