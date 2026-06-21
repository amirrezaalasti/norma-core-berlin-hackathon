"""Live object detection from a laptop webcam using Roboflow serverless inference."""

from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import numpy as np

from norma_vision.env_config import load_env
from norma_vision.manual_workspace_store import load_manual_workspace
from norma_vision.roboflow_detector import RoboflowDetector
from norma_vision.scripts.run_detect_offline import (
    apply_workspace,
    draw_overlay,
    parse_ref_size,
)
from norma_vision.workspace import gripper_tip_position, scale_workspace_to_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Roboflow object detection on a local webcam (press q to quit).",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=int(os.environ.get("NORMA_WEBCAM_INDEX", "0")),
        help="OpenCV camera index (default: 0)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=int(os.environ.get("NORMA_WEBCAM_WIDTH", "640")),
        help="Requested capture width",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=int(os.environ.get("NORMA_WEBCAM_HEIGHT", "480")),
        help="Requested capture height",
    )
    parser.add_argument(
        "--every",
        type=int,
        default=2,
        help="Run inference every N frames (default: 2 — balances speed and freshness)",
    )
    parser.add_argument(
        "--no-workspace-filter",
        action="store_true",
        help="Do not filter detections to manual board region",
    )
    parser.add_argument(
        "--workspace-ref-size",
        default=os.environ.get("NORMA_WORKSPACE_REF_SIZE", "299,224"),
        help="Width,height the manual corners were clicked on",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Optional path to save a snapshot PNG when you press s",
    )
    return parser.parse_args()


def main() -> None:
    load_env()
    args = parse_args()
    ref_size = parse_ref_size(args.workspace_ref_size)

    workspace = load_manual_workspace()
    if workspace is None:
        print(
            "note: no manual workspace — overlay shows pixel boxes only.",
            file=sys.stderr,
        )

    detector = RoboflowDetector()
    print(f"Roboflow model: {detector.config.model_id}", file=sys.stderr)

    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        print(f"error: could not open camera index {args.camera}", file=sys.stderr)
        raise SystemExit(2)

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    latest_detections: list[dict] = []
    backend_used = detector.model_name
    frame_index = 0
    last_inference_ms = 0.0

    try:
        while True:
            ok, bgr = capture.read()
            if not ok:
                print("warning: empty frame", file=sys.stderr)
                break

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            height, width = rgb.shape[:2]
            scaled_workspace = workspace
            if workspace is not None:
                scaled_workspace = scale_workspace_to_image(
                    workspace,
                    width,
                    height,
                    reference_width=ref_size[0],
                    reference_height=ref_size[1],
                )

            if frame_index % max(args.every, 1) == 0:
                started = time.perf_counter()
                detections = detector.detect(rgb)
                last_inference_ms = (time.perf_counter() - started) * 1000.0
                latest_detections = apply_workspace(
                    detections,
                    scaled_workspace,
                    use_filter=not args.no_workspace_filter,
                )

            gripper_tip = gripper_tip_position(scaled_workspace) if scaled_workspace else None
            overlay = draw_overlay(
                rgb,
                latest_detections,
                scaled_workspace,
                gripper_tip,
                backend_used,
            )
            cv2.putText(
                overlay,
                f"infer {last_inference_ms:.0f}ms | {len(latest_detections)} det | q=quit s=save",
                (8, overlay.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow("norma-vision-webcam", overlay)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s") and args.save:
                cv2.imwrite(args.save, overlay)
                print(f"saved {args.save}", file=sys.stderr)

            frame_index += 1
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
