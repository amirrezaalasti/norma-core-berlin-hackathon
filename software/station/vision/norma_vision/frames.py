from __future__ import annotations

import asyncio
import io
import logging
import os
import time
from typing import Any, Protocol

import numpy as np
from PIL import Image

from .paths import setup_import_paths

setup_import_paths()

try:
    from station_py import new_station_client
    from target.gen_python.protobuf.drivers.inferences import normvla
    from target.gen_python.protobuf.drivers.usbvideo import usbvideo
    from target.gen_python.protobuf.station import drivers, inference
except ImportError as exc:
    raise ImportError(
        "Missing generated protobufs or station_py. From repo root run: make protobuf"
    ) from exc

NORMVLA_QUEUE_ID = "inference/normvla"
INFERENCE_STATES_QUEUE = "inference-states"
logger = logging.getLogger("norma-vision")


class FrameReader(Protocol):
    async def connect(self, timeout_s: float = 10.0) -> None: ...

    async def read_rgb(
        self,
        camera_index: int = 0,
        timeout_s: float = 15.0,
        require_fresh: bool = True,
    ) -> tuple[np.ndarray, dict[str, Any]]: ...


def _frame_source_mode() -> str:
    return os.environ.get("NORMA_VISION_FRAME_SOURCE", "inference-states").strip().lower()


def create_frame_reader(host: str) -> FrameReader:
    mode = _frame_source_mode()
    if mode == "normvla":
        return StationFrameReader(host)
    if mode == "auto":
        return CompositeFrameReader(host)
    return InferenceStateFrameReader(host)


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
        self._error_queue = self.client.follow(NORMVLA_QUEUE_ID, self._queue)
        logger.info("Following %s for live camera frames", NORMVLA_QUEUE_ID)

    def _check_follow_error(self) -> None:
        if self._error_queue is None:
            return
        if not self._error_queue.empty():
            raise RuntimeError(str(self._error_queue.get_nowait()))

    async def read_rgb(
        self,
        camera_index: int = 0,
        timeout_s: float = 15.0,
        require_fresh: bool = True,
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
                raise RuntimeError(f"{NORMVLA_QUEUE_ID} follow stream closed")

            frame = normvla.FrameReader(memoryview(bytes(entry.Data)))
            frame_id = bytes(frame.get_global_frame_id())
            if require_fresh and self._last_frame_id is not None and frame_id == self._last_frame_id:
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


def _usbvideo_jpeg_from_envelope(data: bytes) -> bytes | None:
    envelope = usbvideo.RxEnvelopeReader(memoryview(data))
    if envelope.get_type() != usbvideo.RxEnvelopeType.ET_FRAMES:
        return None
    frames = envelope.get_frames()
    if frames is None:
        return None
    frame_data = frames.get_frames_data()
    if frame_data:
        return bytes(frame_data[0])
    linear = frames.get_linear_data()
    return bytes(linear) if linear else None


async def _read_queue_entry(client: Any, queue_id: str, ptr: bytes) -> bytes | None:
    qr = client.read_from_offset(queue_id, ptr, limit=1, step=1, buf_size=1)
    while True:
        try:
            entry = await asyncio.wait_for(qr.data.get(), timeout=5.0)
        except asyncio.TimeoutError:
            return None
        if entry is None:
            return None
        return bytes(entry.Data)


async def rgb_from_inference_state_entry(
    client: Any,
    state_data: bytes,
    camera_index: int = 0,
    video_client: Any | None = None,
) -> tuple[np.ndarray, dict[str, Any]] | None:
    """Resolve usb-video JPEG pointers from an inference-states entry (same path as the viewer)."""
    rx = inference.InferenceRxReader(memoryview(state_data))
    video_entries = [
        entry
        for entry in rx.get_entries()
        if entry.get_type() == drivers.QueueDataType.QDT_USB_VIDEO_FRAMES and entry.get_ptr()
    ]
    if not video_entries:
        return None
    if camera_index < 0 or camera_index >= len(video_entries):
        return None

    target = video_entries[camera_index]
    queue_id = target.get_queue()
    ptr = bytes(target.get_ptr())
    if not queue_id or not ptr:
        return None

    video_data = await _read_queue_entry(video_client or client, queue_id, ptr)
    if video_data is None:
        return None

    jpeg = _usbvideo_jpeg_from_envelope(video_data)
    if jpeg is None:
        return None

    rgb = jpeg_bytes_to_rgb(jpeg)
    height, width = rgb.shape[:2]
    return rgb, {
        "camera_index": camera_index,
        "camera_count": len(video_entries),
        "width": width,
        "height": height,
        "video_queue": queue_id,
        "frame_source": "inference-states",
    }


class InferenceStateFrameReader:
    """Poll inference-states and read usb-video frames (no motor torque required)."""

    def __init__(self, host: str):
        self.host = host
        self.client = None
        self._video_client = None
        self._last_state_id: bytes | None = None

    async def connect(self, timeout_s: float = 10.0) -> None:
        self.client = await new_station_client(self.host, logger)
        self._video_client = await new_station_client(self.host, logger)
        deadline = time.monotonic() + timeout_s
        for station_client in (self.client, self._video_client):
            while not station_client.setup_done:
                if time.monotonic() > deadline:
                    raise TimeoutError(f"Station client setup timed out for {self.host}")
                await asyncio.sleep(0.05)
        logger.info("Polling %s for live usb-video frames", INFERENCE_STATES_QUEUE)

    async def _poll_latest_state(self, timeout_s: float = 2.0) -> tuple[bytes, bytes] | None:
        if self.client is None:
            raise RuntimeError("InferenceStateFrameReader is not connected")

        qr = self.client.read_from_tail(
            INFERENCE_STATES_QUEUE,
            offset=b"\x01",
            limit=1,
            step=1,
            buf_size=1,
        )
        entry = None
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            try:
                item = await asyncio.wait_for(
                    qr.data.get(),
                    timeout=max(0.05, deadline - time.monotonic()),
                )
            except asyncio.TimeoutError:
                break
            if item is None:
                break
            entry = item
            break
        if entry is None:
            if qr.err:
                raise RuntimeError(str(qr.err))
            return None
        return bytes(entry.ID.ID), bytes(entry.Data)

    async def read_rgb(
        self,
        camera_index: int = 0,
        timeout_s: float = 15.0,
        require_fresh: bool = True,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if self.client is None:
            raise RuntimeError("InferenceStateFrameReader is not connected")

        deadline = time.monotonic() + timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    "Timed out waiting for a camera frame from inference-states. "
                    "Is station running with usb-video enabled?"
                )

            polled = await self._poll_latest_state(
                timeout_s=min(2.0, remaining) if require_fresh else min(0.75, remaining),
            )
            if polled is None:
                await asyncio.sleep(min(0.02, remaining))
                continue

            state_id, state_data = polled
            if require_fresh and self._last_state_id is not None and state_id == self._last_state_id:
                await asyncio.sleep(min(0.02, remaining))
                continue

            resolved = await rgb_from_inference_state_entry(
                self.client,
                state_data,
                camera_index=camera_index,
                video_client=self._video_client,
            )
            if resolved is None:
                await asyncio.sleep(min(0.02, remaining))
                continue

            self._last_state_id = state_id
            rgb, meta = resolved
            return rgb, {**meta, "require_fresh": require_fresh}


class CompositeFrameReader:
    """Try inference-states first, then fall back to normvla."""

    def __init__(self, host: str):
        self.host = host
        self._primary = InferenceStateFrameReader(host)
        self._fallback = StationFrameReader(host)
        self._active: FrameReader = self._primary

    async def connect(self, timeout_s: float = 10.0) -> None:
        await self._primary.connect(timeout_s=timeout_s)
        try:
            await self._fallback.connect(timeout_s=timeout_s)
        except Exception as exc:
            logger.warning("normvla frame source unavailable: %s", exc)

    async def read_rgb(
        self,
        camera_index: int = 0,
        timeout_s: float = 15.0,
        require_fresh: bool = True,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        try:
            return await self._primary.read_rgb(
                camera_index=camera_index,
                timeout_s=min(5.0, timeout_s),
                require_fresh=require_fresh,
            )
        except TimeoutError:
            logger.info("Falling back from inference-states to normvla frames")
            return await self._fallback.read_rgb(
                camera_index=camera_index,
                timeout_s=timeout_s,
                require_fresh=require_fresh,
            )


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


_cached_readers: dict[str, FrameReader] = {}
_reader_lock = asyncio.Lock()


async def get_frame_reader(host: str) -> FrameReader:
    async with _reader_lock:
        reader = _cached_readers.get(host)
        if reader is None:
            reader = create_frame_reader(host)
            await reader.connect()
            _cached_readers[host] = reader
        return reader


async def fetch_camera_images(
    host: str,
    camera_index: int = 0,
    timeout_s: float = 5.0,
    require_fresh: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    reader = await get_frame_reader(host)
    return await reader.read_rgb(
        camera_index=camera_index,
        timeout_s=timeout_s,
        require_fresh=require_fresh,
    )
