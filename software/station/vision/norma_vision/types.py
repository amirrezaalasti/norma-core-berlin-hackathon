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

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["obb_xywha"] = [
            self.center_xy[0],
            self.center_xy[1],
            self.size_wh[0],
            self.size_wh[1],
            self.angle_deg,
        ]
        return data
