"""Offline object detection on saved images — no station, camera, or robot required."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from norma_vision.contrast_detector import ContrastDetector
from norma_vision.detector import DEFAULT_CLASSES, DEFAULT_MODEL, ObjectDetector
from norma_vision.env_config import load_env
from norma_vision.manual_workspace_store import load_manual_workspace
from norma_vision.roboflow_detector import RoboflowDetector
from norma_vision.types import Detection
from norma_vision.workspace import (
    enrich_detections_with_workspace,
    filter_detections_in_workspace,
    gripper_tip_position,
    scale_workspace_to_image,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Detect objects on local images using Roboflow, contrast, and/or YOLO. "
            "Uses manual workspace from .norma/manual_workspace.json when present."
        ),
    )
    parser.add_argument(
        "images",
        nargs="+",
        type=Path,
        help="Image file(s) or directories to scan for png/jpg/jpeg",
    )
    parser.add_argument(
        "--backend",
        choices=("roboflow", "contrast", "yolo", "both", "auto"),
        default="auto",
        help=(
            "Detection backend (default: auto). "
            "auto = roboflow then contrast fallback; both = contrast then yolo."
        ),
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("NORMA_VISION_MODEL", DEFAULT_MODEL),
        help=f"YOLO weights (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--classes",
        default=",".join(DEFAULT_CLASSES),
        help="Comma-separated prompts for open-vocabulary YOLO",
    )
    parser.add_argument("--confidence", type=float, default=0.12, help="YOLO confidence threshold")
    parser.add_argument("--device", default=None, help="YOLO device: cpu, mps, cuda:0")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Write annotated overlay PNGs here",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Write combined JSON results to this file (also prints to stdout)",
    )
    parser.add_argument(
        "--no-workspace-filter",
        action="store_true",
        help="Do not filter detections to manual board region",
    )
    parser.add_argument(
        "--workspace-ref-size",
        default=os.environ.get("NORMA_WORKSPACE_REF_SIZE", "299,224"),
        help="Width,height the manual corners were clicked on (default: 299,224 USB stream)",
    )
    return parser.parse_args()


def load_image_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def collect_images(paths: list[Path]) -> list[Path]:
    images: list[Path] = []
    for path in paths:
        if path.is_dir():
            for pattern in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
                images.extend(sorted(path.glob(pattern)))
        elif path.is_file():
            images.append(path)
        else:
            print(f"warning: skipping missing path {path}", file=sys.stderr)
    return images


def detect_contrast(rgb: np.ndarray, classes: list[str]) -> tuple[list[Detection], str]:
    detector = ContrastDetector(classes=classes or ["block"])
    return detector.detect(rgb), "local-contrast"


def detect_yolo(
    rgb: np.ndarray,
    *,
    model: str,
    classes: list[str],
    confidence: float,
    device: str | None,
) -> tuple[list[Detection], str]:
    detector = ObjectDetector(
        model_name=model,
        classes=classes,
        confidence=confidence,
        device=device,
    )
    return detector.detect(rgb), model


def detect_roboflow(
    rgb: np.ndarray,
    detector: RoboflowDetector | None,
) -> tuple[list[Detection], str, RoboflowDetector]:
    if detector is None:
        detector = RoboflowDetector()
    return detector.detect(rgb), detector.model_name, detector


def apply_workspace(
    detections: list[Detection],
    workspace,
    *,
    use_filter: bool,
) -> list[dict]:
    if workspace is None:
        return [item.to_dict() for item in detections]

    dicts = enrich_detections_with_workspace([item.to_dict() for item in detections], workspace)
    if use_filter and workspace.calibration_source == "manual":
        dicts = filter_detections_in_workspace(dicts, workspace)
    return dicts


def draw_overlay(
    rgb: np.ndarray,
    detections: list[dict],
    workspace,
    gripper_tip: dict | None,
    backend: str,
) -> np.ndarray:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    if workspace is not None:
        corners = np.array(workspace.corners_xy, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(bgr, [corners], True, (255, 180, 60), 2, cv2.LINE_AA)
        labels = ["TL", "TR", "BR", "BL"]
        for (x, y), label in zip(workspace.corners_xy, labels, strict=True):
            cv2.circle(bgr, (int(x), int(y)), 5, (60, 180, 255), -1, cv2.LINE_AA)
            cv2.putText(
                bgr,
                label,
                (int(x) + 6, int(y) - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (60, 180, 255),
                1,
                cv2.LINE_AA,
            )

        origin = workspace.origin_xy or workspace.center_xy
        if origin is not None:
            ox, oy = int(origin[0]), int(origin[1])
            cv2.circle(bgr, (ox, oy), 7, (255, 60, 60), -1, cv2.LINE_AA)
            cv2.circle(bgr, (ox, oy), 7, (255, 255, 255), 1, cv2.LINE_AA)
            if gripper_tip and gripper_tip.get("board_xy"):
                bx, by = gripper_tip["board_xy"]
                tip_label = f"tip ({bx:.2f},{by:.2f})"
                cv2.putText(
                    bgr,
                    tip_label,
                    (ox + 10, oy + 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (180, 220, 255),
                    1,
                    cv2.LINE_AA,
                )

    for index, det in enumerate(detections):
        x1, y1, x2, y2 = (int(v) for v in det["bbox_xyxy"])
        color = (80, 220, 120) if index == 0 else (80, 180, 255)
        cv2.rectangle(bgr, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

        parts = [det["class_name"], f"{det['confidence'] * 100:.0f}%"]
        if det.get("board_xy"):
            bx, by = det["board_xy"]
            parts.append(f"board({bx:.2f},{by:.2f})")
        if det.get("offset_xy") is not None and det.get("distance") is not None:
            dx, dy = det["offset_xy"]
            parts.append(f"off({dx:.0f},{dy:.0f}) d={det['distance']:.0f}mm")
        label = " | ".join(parts)
        cv2.putText(
            bgr,
            label,
            (x1, max(12, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
            cv2.LINE_AA,
        )

        if workspace is not None and workspace.origin_xy:
            ox, oy = (int(v) for v in workspace.origin_xy)
            cx, cy = (int(v) for v in det["center_xy"])
            cv2.line(bgr, (ox, oy), (cx, cy), (60, 220, 255), 1, cv2.LINE_AA)

    header = f"{backend} | {len(detections)} det"
    cv2.rectangle(bgr, (4, 4), (min(bgr.shape[1] - 4, 420), 24), (20, 20, 20), -1)
    cv2.putText(
        bgr,
        header,
        (8, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )
    return bgr


def parse_ref_size(raw: str) -> tuple[float, float]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError(f"expected WIDTH,HEIGHT got {raw!r}")
    return float(parts[0]), float(parts[1])


def process_image(
    path: Path,
    args: argparse.Namespace,
    classes: list[str],
    workspace,
    yolo_detector: ObjectDetector | None,
    roboflow_detector: RoboflowDetector | None,
    ref_size: tuple[float, float],
) -> dict:
    rgb = load_image_rgb(path)
    height, width = rgb.shape[:2]

    if workspace is not None:
        workspace = scale_workspace_to_image(
            workspace,
            width,
            height,
            reference_width=ref_size[0],
            reference_height=ref_size[1],
        )
    detections: list[Detection] = []
    backend_used = "none"

    if args.backend in ("roboflow", "auto"):
        detections, backend_used, roboflow_detector = detect_roboflow(rgb, roboflow_detector)
    if not detections and args.backend in ("contrast", "both", "auto"):
        detections, backend_used = detect_contrast(rgb, classes)
    if not detections and args.backend in ("yolo", "both"):
        if yolo_detector is None:
            yolo_detector = ObjectDetector(
                model_name=args.model,
                classes=classes,
                confidence=args.confidence,
                device=args.device,
            )
        detections, backend_used = detect_yolo(
            rgb,
            model=args.model,
            classes=classes,
            confidence=args.confidence,
            device=args.device,
        )

    gripper_tip = gripper_tip_position(workspace) if workspace else None
    detection_dicts = apply_workspace(
        detections,
        workspace,
        use_filter=not args.no_workspace_filter,
    )

    result: dict = {
        "image": str(path.resolve()),
        "width": rgb.shape[1],
        "height": rgb.shape[0],
        "backend": backend_used,
        "detection_count": len(detection_dicts),
        "detections": detection_dicts,
        "workspace": workspace.to_dict() if workspace else None,
        "gripper_tip": gripper_tip,
    }

    if args.out_dir is not None:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        overlay = draw_overlay(rgb, detection_dicts, workspace, gripper_tip, backend_used)
        out_path = args.out_dir / f"{path.stem}_detections.png"
        cv2.imwrite(str(out_path), overlay)
        result["overlay_path"] = str(out_path.resolve())

    return result


def main() -> None:
    load_env()
    args = parse_args()
    images = collect_images(args.images)
    if not images:
        print("error: no images found", file=sys.stderr)
        raise SystemExit(2)

    classes = [item.strip() for item in args.classes.split(",") if item.strip()]
    workspace = load_manual_workspace()
    ref_size = parse_ref_size(args.workspace_ref_size)
    if workspace is None:
        print(
            "note: no manual workspace — detections will be in pixel coords only. "
            "Add .norma/manual_workspace.json or calibrate in station viewer.",
            file=sys.stderr,
        )

    yolo_detector: ObjectDetector | None = None
    if args.backend == "yolo":
        yolo_detector = ObjectDetector(
            model_name=args.model,
            classes=classes,
            confidence=args.confidence,
            device=args.device,
        )

    roboflow_detector: RoboflowDetector | None = None
    if args.backend in ("roboflow", "auto"):
        roboflow_detector = RoboflowDetector()

    results = [
        process_image(path, args, classes, workspace, yolo_detector, roboflow_detector, ref_size)
        for path in images
    ]
    payload = {"results": results, "image_count": len(results)}

    text = json.dumps(payload, indent=2)
    print(text)

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text)


if __name__ == "__main__":
    main()
