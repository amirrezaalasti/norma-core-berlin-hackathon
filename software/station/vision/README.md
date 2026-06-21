# NormaCore Station Vision

> Part of the [NormaCore Berlin Hackathon](../../../README.md) project.

Local object detection for the Station camera overlay and optional arm control via SmolVLA / MCP pick.

## Camera overlay (browser)

The Station viewer runs **local contrast vision** in the browser:

1. Switch to **camera view**
2. Click the **scan/search icon** in the HUD

This finds dark blocks on a bright board and shows gripper-relative offsets from the **blue dot** at the gripper tip.

Rebuild after UI changes:

```bash
cd software/station/bin/station && make client
```

### Roboflow detection (recommended)

Copy `.env.example` to the repo root as `.env` and set `ROBOFLOW_API_KEY` from
[Roboflow settings](https://app.roboflow.com/settings/api).

Default model `yolov8s-640` is a fast COCO detector on Roboflow serverless. Use
`yolov8n-640` for maximum speed or point `ROBOFLOW_MODEL_ID` at your own Universe
project (`workspace/model/version`).

```bash
cd software/station/vision
uv sync --extra roboflow

# Saved images (default: roboflow, then contrast if empty)
uv run norma-vision-offline ../../../images/image.png --out-dir /tmp/vision

# Roboflow only
uv run norma-vision-offline ../../../images/image.png --backend roboflow --out-dir /tmp/vision

# Contrast fallback when Roboflow finds nothing
uv run norma-vision-offline ../../../images/image.png --backend auto --out-dir /tmp/vision

# Laptop webcam (press q to quit, s to save snapshot if --save set)
uv run norma-vision-webcam --camera 0
```

Writes annotated PNGs (`--out-dir`) and JSON with `board_xy` / offset when `.norma/manual_workspace.json` exists.

### Offline detection (no robot or camera)

Iterate on saved screenshots without the station:

```bash
cd software/station/vision

uv run norma-vision-offline \
  ../../../images/Screenshot*.png \
  --out-dir ../../../images/debug_calibration/offline_run

# Contrast only (best for dark blocks on white board)
uv run norma-vision-offline ../../../images/image.png --backend contrast --out-dir /tmp/vision

# Local YOLO only
uv run norma-vision-offline ../../../images/image.png --backend yolo --confidence 0.1
```

### Coordinate system (browser)

| Element | Role |
|---------|------|
| **Blue dot** | Gripper tip = origin `(0, 0)` |
| **+x / +y axes** | Board-plane directions (perspective corrected) |
| **Object label** | `x`, `y`, `d` = offset and distance from gripper |

Browser mode uses **white-board edge detection** → distances are in **board-plane pixels** (good for relative motion, not true mm).

---

## Camera intrinsics / extrinsics (millimeter accuracy)

When `images/intrinsics.json` and `images/extrinsics.json` are present at the repo root,
Python vision and the browser overlay automatically:

1. **Undistort** detection pixels (not the full frame) before ray-casting to the board plane
2. **Ray-cast** pixels to the board plane using `T_cam2world` for offsets in **mm**
3. **MCP pick** uses mm joint scales (`DEFAULT_PICK_SCALES_MM`) when `calibration_source` is `camera`

Paths can be overridden:

```bash
export NORMA_INTRINSICS_PATH=/path/to/intrinsics.json
export NORMA_EXTRINSICS_PATH=/path/to/extrinsics.json
export NORMA_INTRINSICS_VARIANT=aligned   # or unaligned
export NORMA_BOARD_PLANE_Z_MM=0
```

The live vision API exposes calibration at `GET /calibration/camera` (port 8890).

MCP `detect_workspace_objects` reuses a cached station frame reader and accepts the latest
frame immediately (`require_fresh=false`) so it returns in ~1s instead of waiting for a new
inference-states entry.

---

## AprilTags (millimeter accuracy)

For **real distances in mm**, use four corner **AprilTag 36h11** markers and the Python vision path.

### Why AprilTags?

```
Camera image (distorted)  →  4 tag centers  →  Homography  →  Board plane in mm
Blue gripper dot          →  origin (0,0) in gripper frame
Object center             →  offset (dx, dy) and distance in mm
```

The tags define the **scale and shape** of the board. The blue dot defines **where the gripper is** relative to objects.

### Setup

1. Print four **tagStandard41h12** tags ([AprilRobotics default](https://github.com/AprilRobotics/apriltag#choosing-a-tag-family)) — or **tag16h5** if you already printed those.
2. Place one near each corner of the white foam board.
3. Measure the board and tag placement:

```bash
# Default tries tagStandard41h12 first, then tag16h5, etc.
# export NORMA_APRILTAG_FAMILY=tagStandard41h12
export NORMA_WORKSPACE_MODE=auto
export NORMA_BOARD_WIDTH_MM=280
export NORMA_BOARD_HEIGHT_MM=200
export NORMA_TAG_INSET_MM=25
# Optional if you know tag IDs (TL, TR, BR, BL):
# export NORMA_APRILTAG_IDS=4,8,14,16
```

4. Arm at **initialized pose** → `save_home_pose` (MCP). Origin is the center of the four tags.

### Run Python vision with AprilTags

```bash
uv run --project software/station/vision norma-vision-live \
  --backend contrast \
  --station-host localhost:8888
```

Point the viewer at port 8890 or use MCP `detect_workspace_objects`. Status should show **`apriltag mm`**.

Detection uses `pupil-apriltags` (supports `tagStandard41h12`). Optionally install the official
[AprilRobotics apriltag](https://github.com/AprilRobotics/apriltag#python) bindings for the same API;
on macOS build from source (`cmake -B build -DCMAKE_POLICY_VERSION_MINIMUM=3.5 && cmake --build build`).
The PyPI `apriltag` wheel often fails to build on newer CMake.

### Pick workflow (MCP)

```
1. save_home_pose              # arm at init pose, blue dot visible, tags visible
2. detect_workspace_objects    # verify offset e.g. x:-23 y:33 d:40mm
3. pick_nearest_object         # move from home by offset, close gripper
```

Joint motion is **home pose + scaled (dx, dy)**. Tune with `NORMA_PICK_SCALES` if the arm undershoots.

---

## SmolVLA — learned arm control

SmolVLA outputs **joint motions** from images, not Cartesian targets. Use it when you have a fine-tuned policy:

```bash
cd software/ai/smolvla_py
uv run python scripts/run_policy.py \
  --checkpoint lerobot/smolvla_base \
  --task "pick up the black block" \
  --bus-serial YOUR_BUS_ID \
  --server localhost
```

| Layer | Tool | Purpose |
|-------|------|---------|
| Where is the object? | Vision overlay / AprilTags | Offset from gripper |
| How should the arm move? | MCP pick or SmolVLA | Joint commands |

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Overlay shows `board only` | Blue gripper marker not visible |
| Distances in px, need mm | Run Python vision with AprilTags + measure board |
| AprilTags not detected | Match family to print (`tagStandard41h12` or `tag16h5`), improve lighting, set `NORMA_APRILTAG_IDS` |
| Pick misses target | Tune `NORMA_PICK_SCALES`, re-run `save_home_pose` at init pose |
| Arm doesn't move | Enable torque, calibrate motors, use MCP not overlay alone |
