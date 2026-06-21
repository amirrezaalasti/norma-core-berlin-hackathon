from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


def grid_dimensions() -> tuple[int, int]:
    cols = max(1, int(os.environ.get("NORMA_BOARD_GRID_COLS", "5")))
    rows = max(1, int(os.environ.get("NORMA_BOARD_GRID_ROWS", "3")))
    return cols, rows


@dataclass(frozen=True)
class WorkspaceSquareInfo:
    square_id: int
    square_col: int
    square_row: int
    square_center_board_xy: tuple[float, float]
    square_local_xy: tuple[float, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "square_id": self.square_id,
            "square_col": self.square_col,
            "square_row": self.square_row,
            "square_center_board_xy": [
                round(self.square_center_board_xy[0], 4),
                round(self.square_center_board_xy[1], 4),
            ],
            "square_local_xy": [
                round(self.square_local_xy[0], 4),
                round(self.square_local_xy[1], 4),
            ],
        }


def square_info_from_board_xy(
    board_xy: tuple[float, float],
    *,
    cols: int | None = None,
    rows: int | None = None,
) -> WorkspaceSquareInfo:
    if cols is None or rows is None:
        cols, rows = grid_dimensions()

    u, v = board_xy
    col = min(cols - 1, max(0, int(u * cols)))
    row = min(rows - 1, max(0, int(v * rows)))
    center_u = (col + 0.5) / cols
    center_v = (row + 0.5) / rows
    return WorkspaceSquareInfo(
        square_id=row * cols + col + 1,
        square_col=col,
        square_row=row,
        square_center_board_xy=(center_u, center_v),
        square_local_xy=((u - center_u) * cols, (v - center_v) * rows),
    )
