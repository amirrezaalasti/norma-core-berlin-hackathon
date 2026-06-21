from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class Detection:
    class_name: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]
    center_xy: tuple[float, float]
    size_wh: tuple[float, float]
    angle_deg: float
    board_xy: tuple[float, float] | None = None
    offset_xy: tuple[float, float] | None = None
    distance: float | None = None
    square_id: int | None = None
    square_col: int | None = None
    square_row: int | None = None
    square_center_board_xy: tuple[float, float] | None = None
    square_local_xy: tuple[float, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["obb_xywha"] = [
            self.center_xy[0],
            self.center_xy[1],
            self.size_wh[0],
            self.size_wh[1],
            self.angle_deg,
        ]
        if self.board_xy is None:
            data.pop("board_xy", None)
        if self.offset_xy is None:
            data.pop("offset_xy", None)
        if self.distance is None:
            data.pop("distance", None)
        if self.square_id is None:
            data.pop("square_id", None)
            data.pop("square_col", None)
            data.pop("square_row", None)
            data.pop("square_center_board_xy", None)
            data.pop("square_local_xy", None)
        return data
