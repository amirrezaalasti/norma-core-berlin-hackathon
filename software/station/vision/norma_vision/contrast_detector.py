from __future__ import annotations

import numpy as np

from .contrast_fallback import detect_dark_objects
from .types import Detection

DEFAULT_CLASSES = ["black block"]


class ContrastDetector:
    """Fast local vision for dark objects on bright surfaces — no ML weights."""

    def __init__(
        self,
        classes: list[str] | None = None,
        **_kwargs: object,
    ):
        self.classes = list(classes or DEFAULT_CLASSES)
        self.model_name = "local-contrast"

    def detect(self, image_rgb: np.ndarray) -> list[Detection]:
        class_name = self.classes[0] if self.classes else "black block"
        return detect_dark_objects(image_rgb, class_name=class_name)
