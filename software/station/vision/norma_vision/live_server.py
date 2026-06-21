from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .detector import DEFAULT_CLASSES, DEFAULT_MODEL, ObjectDetector
from .contrast_detector import ContrastDetector
from .roboflow_workspace_detector import RoboflowWorkspaceDetector
from .frames import create_frame_reader

logger = logging.getLogger("norma-vision-live")

_latest: dict[str, Any] = {
    "width": 0,
    "height": 0,
    "camera_index": 0,
    "model": DEFAULT_MODEL,
    "classes": DEFAULT_CLASSES,
    "detection_count": 0,
    "detections": [],
    "inference_fps": 0.0,
    "updated_at_ms": 0,
    "error": "Starting detection loop...",
}
_latest_lock = threading.Lock()


def get_latest_snapshot() -> dict[str, Any]:
    with _latest_lock:
        return dict(_latest)


def set_latest_snapshot(payload: dict[str, Any]) -> None:
    with _latest_lock:
        _latest.clear()
        _latest.update(payload)


class VisionRequestHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        logger.debug(format, *args)

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:
        if self.path in ("/", "/health"):
            self._send_json(200, {"status": "ok"})
            return
        if self.path == "/latest":
            self._send_json(200, get_latest_snapshot())
            return
        if self.path == "/calibration/manual":
            from .manual_workspace_store import load_manual_workspace

            workspace = load_manual_workspace()
            if workspace is None:
                self._send_json(404, {"error": "No manual calibration saved"})
                return
            self._send_json(200, {"workspace": workspace.to_dict(), "ready": True})
            return
        if self.path == "/calibration/camera":
            from .camera_calibration import calibration_payload_for_api

            payload = calibration_payload_for_api()
            if payload is None:
                self._send_json(404, {"error": "No camera calibration found"})
                return
            self._send_json(200, payload)
            return
        self._send_json(404, {"error": "Not found"})

    def do_POST(self) -> None:
        if self.path != "/calibration/manual":
            self._send_json(404, {"error": "Not found"})
            return

        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length > 0 else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            self._send_json(400, {"error": "Invalid JSON"})
            return

        from .manual_workspace_store import manual_workspace_ready, save_manual_workspace

        try:
            workspace = save_manual_workspace(payload)
        except ValueError as exc:
            self._send_json(400, {"error": str(exc)})
            return

        self._send_json(
            200,
            {
                "saved": True,
                "ready": manual_workspace_ready(payload),
                "workspace": workspace.to_dict(),
            },
        )

    def do_DELETE(self) -> None:
        if self.path != "/calibration/manual":
            self._send_json(404, {"error": "Not found"})
            return

        from .manual_workspace_store import clear_manual_workspace

        clear_manual_workspace()
        self._send_json(200, {"cleared": True})


async def detection_loop(
    host: str,
    detector: ContrastDetector | ObjectDetector | RoboflowWorkspaceDetector,
    camera_index: int,
    target_fps: float,
) -> None:
    reader = create_frame_reader(host)
    await reader.connect()

    frame_times: list[float] = []
    loop = asyncio.get_running_loop()

    while True:
        started = time.perf_counter()
        try:
            rgb, meta = await reader.read_rgb(camera_index=camera_index)
            detections = await loop.run_in_executor(None, detector.detect, rgb)
            now_ms = int(time.time() * 1000)

            frame_times.append(started)
            frame_times = [stamp for stamp in frame_times if started - stamp <= 1.0]
            inference_fps = len(frame_times)

            workspace = getattr(detector, "last_workspace", None)
            calibration = getattr(detector, "last_calibration", None)
            gripper_tip = getattr(detector, "last_gripper_tip", None)

            set_latest_snapshot(
                {
                    **meta,
                    "model": detector.model_name,
                    "classes": detector.classes,
                    "detection_count": len(detections),
                    "detections": [item.to_dict() for item in detections],
                    "workspace": workspace.to_dict() if workspace is not None else None,
                    "gripper_tip": gripper_tip,
                    "camera_calibration": calibration.to_dict() if calibration is not None else None,
                    "inference_fps": float(inference_fps),
                    "updated_at_ms": now_ms,
                    "error": None,
                }
            )
        except Exception as exc:
            logger.exception("Detection loop error")
            snapshot = get_latest_snapshot()
            snapshot["error"] = str(exc)
            snapshot["updated_at_ms"] = int(time.time() * 1000)
            set_latest_snapshot(snapshot)

        elapsed = time.perf_counter() - started
        sleep_s = max(0.0, (1.0 / target_fps) - elapsed)
        await asyncio.sleep(sleep_s)


def run_live_server(
    host: str,
    station_host: str,
    port: int,
    backend: str,
    model_name: str,
    classes: list[str],
    confidence: float,
    camera_index: int,
    target_fps: float,
    device: str | None,
    use_contrast_fallback: bool = True,
) -> None:
    logging.basicConfig(level=logging.INFO)

    if backend == "roboflow":
        detector: ContrastDetector | ObjectDetector | RoboflowWorkspaceDetector = (
            RoboflowWorkspaceDetector()
        )
    elif backend == "contrast":
        detector = ContrastDetector(classes=classes)
    else:
        detector = ObjectDetector(
            model_name=model_name,
            classes=classes,
            confidence=confidence,
            device=device or os.environ.get("NORMA_VISION_DEVICE"),
            use_contrast_fallback=use_contrast_fallback,
        )

    httpd = ThreadingHTTPServer((host, port), VisionRequestHandler)
    server_thread = threading.Thread(
        target=httpd.serve_forever,
        name="norma-vision-http",
        daemon=True,
    )
    server_thread.start()
    logger.info("Vision overlay API listening on http://%s:%s/latest", host, port)

    asyncio.run(
        detection_loop(
            host=station_host,
            detector=detector,
            camera_index=camera_index,
            target_fps=target_fps,
        )
    )
