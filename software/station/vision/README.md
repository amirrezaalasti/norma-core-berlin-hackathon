# NormaCore Station Vision

Pretrained object detection for robot pick-and-place — **no custom training**.

Uses [Ultralytics YOLOE](https://docs.ultralytics.com/models/yoloe/) by default: open-vocabulary detection with text prompts plus instance segmentation. Segmentation masks are converted to oriented bounding boxes `[x, y, w, h, θ]` for gripper alignment.

## Prerequisites

From the repo root:

```bash
make protobuf
```

Station must be running with `usb-video` enabled (see `software/station/bin/station/station.yaml`).

## Install

```bash
uv sync --project software/station/vision
```

Model weights are **not** in git. Ultralytics downloads them on first run (~20–50 MB for `yoloe-11s-seg.pt`), or place `yoloe-11s-seg.pt` in the repo root manually.

Do not commit `*.pt` or other weight files.

## Detect on a live camera frame

```bash
uv run --project software/station/vision norma-vision-detect \
  --source station \
  --host localhost:8888 \
  --classes "cube,mug,rectangular box"
```

## Detect on a local image

```bash
uv run --project software/station/vision norma-vision-detect \
  --source file \
  --image path/to/photo.jpg \
  --classes "yellow cube,coffee mug,box"
```

## Model options

| Model | Type | When to use |
|-------|------|-------------|
| `yoloe-11s-seg.pt` (default) | Open vocab + seg | Best zero-shot; gives rotation from mask |
| `yoloe-11m-seg.pt` | Open vocab + seg | More accurate, slower |
| `yolov8s-worldv2.pt` | Open vocab | Lighter, no segmentation |
| `yolo11n.pt` | COCO fixed classes | Only detects 80 COCO categories (`cup`, `bottle`, …) |

Set via `--model` or `NORMA_VISION_MODEL`.

## Output

Each detection includes:

- `class_name`, `confidence`
- `bbox_xyxy` — axis-aligned box in pixels
- `center_xy`, `size_wh`, `angle_deg` — oriented box from mask (or axis-aligned fallback)
- `obb_xywha` — `[x, y, w, h, angle_deg]`

Next step for grasping: camera calibration + hand-eye transform to map pixel `(x, y)` and `angle_deg` into the arm workspace, then IK to joint goals via Station MCP.

## MCP integration

With vision dependencies installed on the MCP project:

```bash
uv sync --project software/station/mcp --extra vision
```

Reload MCP in Cursor, then use the `detect_objects` tool (same text prompts as `--classes`).
