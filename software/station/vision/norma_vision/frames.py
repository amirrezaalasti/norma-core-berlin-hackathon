from __future__ import annotations

import asyncio
import io
import logging
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


def jpeg_bytes_to_rgb(jpeg: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(jpeg)) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


async def fetch_normvla_frame(
    host: str,
    timeout_s: float = 5.0,
) -> normvla.FrameReader:
    client = await new_station_client(host, logger)
    qr = client.read_from_tail(QUEUE_ID, offset=b"\x00", limit=1, step=1, buf_size=1)
    entry = await asyncio.wait_for(qr.data.get(), timeout=timeout_s)
    if entry is None:
        raise RuntimeError(f"{QUEUE_ID} closed without delivering a frame ({qr.err})")
    return normvla.FrameReader(memoryview(bytes(entry.Data)))


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
    frame = await fetch_normvla_frame(host, timeout_s=timeout_s)
    images = frame_images_rgb(frame)
    if camera_index < 0 or camera_index >= len(images):
        raise IndexError(
            f"camera_index {camera_index} out of range; frame has {len(images)} image(s)"
        )

    rgb = images[camera_index]
    height, width = rgb.shape[:2]
    meta = {
        "camera_index": camera_index,
        "camera_count": len(images),
        "width": width,
        "height": height,
        "global_frame_id": bytes(frame.get_global_frame_id()).hex(),
    }
    return rgb, meta
