from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from norma_vision.detector import COCO_MODEL, DEFAULT_CLASSES, DEFAULT_MODEL, ObjectDetector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect objects with a pretrained YOLOE/YOLO model (no training)."
    )
    parser.add_argument(
        "--source",
        choices=("station", "file"),
        default="station",
        help="Read a live camera frame from station or a local image file.",
    )
    parser.add_argument("--image", type=Path, help="Image path when --source=file")
    parser.add_argument(
        "--host",
        default=os.environ.get("STATION_HOST", "localhost:8888"),
        help="Station TCP host when --source=station",
    )
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument(
        "--model",
        default=os.environ.get("NORMA_VISION_MODEL", DEFAULT_MODEL),
        help=f"Ultralytics weights (default: {DEFAULT_MODEL}). "
        f"Open-vocab: yoloe-11s-seg.pt, yolov8s-worldv2.pt. COCO: {COCO_MODEL}",
    )
    parser.add_argument(
        "--classes",
        default=",".join(DEFAULT_CLASSES),
        help="Comma-separated text prompts for open-vocabulary models",
    )
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--device", default=None, help="e.g. cpu, mps, cuda:0")
    return parser.parse_args()


def load_image_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


async def run_station(args: argparse.Namespace, detector: ObjectDetector) -> dict:
    from norma_vision.frames import fetch_camera_images

    rgb, meta = await fetch_camera_images(
        host=args.host,
        camera_index=args.camera_index,
    )
    detections = detector.detect(rgb)
    return {
        **meta,
        "model": args.model,
        "classes": detector.classes,
        "detection_count": len(detections),
        "detections": [item.to_dict() for item in detections],
    }


def main() -> None:
    args = parse_args()

    if args.source == "file":
        if args.image is None:
            print("error: --image is required when --source=file", file=sys.stderr)
            raise SystemExit(2)
        rgb = load_image_rgb(args.image)
        meta = {
            "source": str(args.image),
            "width": rgb.shape[1],
            "height": rgb.shape[0],
        }
    else:
        meta = None

    classes = [item.strip() for item in args.classes.split(",") if item.strip()]
    detector = ObjectDetector(
        model_name=args.model,
        classes=classes,
        confidence=args.confidence,
        device=args.device,
    )

    if args.source == "station":
        payload = asyncio.run(run_station(args, detector))
    else:
        detections = detector.detect(rgb)
        payload = {
            **meta,
            "model": args.model,
            "classes": detector.classes,
            "detection_count": len(detections),
            "detections": [item.to_dict() for item in detections],
        }

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
