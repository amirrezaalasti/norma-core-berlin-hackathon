from __future__ import annotations

import asyncio
import io
import logging
import time
from typing import Any

import numpy as np
from PIL import Image

from .paths import setup_import_paths

setup_import_paths()

try:
    from station_py import new_station_client
    from target.gen_python.protobuf.drivers.inferences import normvla
except ImportError as exc:
    raise ImportError(
        "Missing generated protobufs or station_py. From repo root run: make protobuf"
    ) from exc

QUEUE_ID = "inference/normvla"
logger = logging.getLogger("norma-vision")


class StationFrameReader:
    """Persistent NormFS client that follows inference/normvla."""

    def __init__(self, host: str):
        self.host = host
        self.client = None
        self._queue: asyncio.Queue | None = None
        self._error_queue: asyncio.Queue | None = None
        self._last_frame_id: bytes | None = None
        self._last_frame: normvla.FrameReader | None = None

    async def connect(self, timeout_s: float = 10.0) -> None:
        self.client = await new_station_client(self.host, logger)
        deadline = time.monotonic() + timeout_s
        while not self.client.setup_done:
            if time.monotonic() > deadline:
                raise TimeoutError(f"Station client setup timed out for {self.host}")
            await asyncio.sleep(0.05)

        self._queue = asyncio.Queue()
        self._error_queue = self.client.follow(QUEUE_ID, self._queue)
        logger.info("Following %s for live camera frames", QUEUE_ID)

    def _check_follow_error(self) -> None:
        if self._error_queue is None:
            return
        if not self._error_queue.empty():
            raise RuntimeError(str(self._error_queue.get_nowait()))

    async def read_rgb(
        self,
        camera_index: int = 0,
        timeout_s: float = 15.0,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if self.client is None or self._queue is None:
            raise RuntimeError("StationFrameReader is not connected")

        deadline = time.monotonic() + timeout_s
        while True:
            self._check_follow_error()

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    "Timed out waiting for a fresh normvla frame. "
                    "Is station running with usb-video and inference/normvla enabled?"
                )

            try:
                entry = await asyncio.wait_for(self._queue.get(), timeout=min(1.0, remaining))
            except asyncio.TimeoutError:
                continue

            if entry is None:
                self._check_follow_error()
                raise RuntimeError(f"{QUEUE_ID} follow stream closed")

            frame = normvla.FrameReader(memoryview(bytes(entry.Data)))
            frame_id = bytes(frame.get_global_frame_id())
            if self._last_frame_id is not None and frame_id == self._last_frame_id:
                continue

            self._last_frame_id = frame_id
            self._last_frame = frame
            images = frame_images_rgb(frame)
            if camera_index < 0 or camera_index >= len(images):
                raise IndexError(
                    f"camera_index {camera_index} out of range; "
                    f"frame has {len(images)} image(s)"
                )

            rgb = images[camera_index]
            height, width = rgb.shape[:2]
            return rgb, {
                "camera_index": camera_index,
                "camera_count": len(images),
                "width": width,
                "height": height,
                "global_frame_id": frame_id.hex(),
            }


def jpeg_bytes_to_rgb(jpeg: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(jpeg)) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


async def fetch_normvla_frame(
    host: str,
    timeout_s: float = 5.0,
) -> normvla.FrameReader:
    reader = StationFrameReader(host)
    await reader.connect(timeout_s=timeout_s)
    await reader.read_rgb(timeout_s=timeout_s)
    if reader._last_frame is None:
        raise RuntimeError("Failed to capture normvla frame")
    return reader._last_frame


def frame_images_rgb(frame: normvla.FrameReader) -> list[np.ndarray]:
    images = frame.get_images() or []
    if not images:
        raise RuntimeError("Frame has no camera images. Is usb-video enabled in station.yaml?")
    return [jpeg_bytes_to_rgb(bytes(image.get_jpeg())) for image in images]


async def fetch_camera_images(
    host: str,
    camera_index: int = 0,
    timeout_s: float = 5.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    reader = StationFrameReader(host)
    await reader.connect(timeout_s=timeout_s)
    return await reader.read_rgb(camera_index=camera_index, timeout_s=timeout_s)
