from __future__ import annotations

import sys
from pathlib import Path

# software/station/mcp/norma_station_mcp/paths.py -> repo root is 4 levels up
REPO_ROOT = Path(__file__).resolve().parents[4]
STATION_PY_ROOT = REPO_ROOT / "software" / "station" / "shared"


def setup_import_paths() -> None:
    """Make generated protobufs and station_py importable."""
    for path in (REPO_ROOT, STATION_PY_ROOT):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
