from __future__ import annotations

import argparse
import os

from norma_vision.detector import DEFAULT_CLASSES, DEFAULT_MODEL
from norma_vision.live_server import run_live_server


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run live object detection and expose overlays for Station camera view."
    )
    parser.add_argument(
        "--bind",
        default=os.environ.get("NORMA_VISION_BIND", "127.0.0.1"),
        help="HTTP bind address for /latest overlay API",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("NORMA_VISION_PORT", "8890")),
        help="HTTP port for overlay API (default: 8890)",
    )
    parser.add_argument(
        "--station-host",
        default=os.environ.get("STATION_HOST", "localhost:8888"),
        help="NormaCore Station TCP host",
    )
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument(
        "--backend",
        choices=("yolo", "contrast"),
        default="yolo",
        help="Detection backend: 'yolo' (YOLO model, default) or 'contrast' (local dark-blob only)",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("NORMA_VISION_MODEL", DEFAULT_MODEL),
    )
    parser.add_argument(
        "--classes",
        default=",".join(DEFAULT_CLASSES),
        help="Comma-separated text prompts",
    )
    parser.add_argument("--confidence", type=float, default=0.1)
    parser.add_argument(
        "--no-contrast-fallback",
        action="store_true",
        help="Disable dark-blob fallback when YOLO finds nothing",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=float(os.environ.get("NORMA_VISION_FPS", "5")),
        help="Target inference rate (default: 5 FPS)",
    )
    parser.add_argument("--device", default=None, help="cpu, mps, cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    classes = [item.strip() for item in args.classes.split(",") if item.strip()]
    run_live_server(
        host=args.bind,
        station_host=args.station_host,
        port=args.port,
        backend=args.backend,
        model_name=args.model,
        classes=classes,
        confidence=args.confidence,
        camera_index=args.camera_index,
        target_fps=args.fps,
        device=args.device,
        use_contrast_fallback=not args.no_contrast_fallback,
    )


if __name__ == "__main__":
    main()

