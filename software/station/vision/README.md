# NormaCore Station Vision

Local object detection for the Station camera overlay and optional arm control via SmolVLA.

## Camera overlay (no extra server)

The Station viewer runs **local contrast vision** directly in the browser:

1. Switch to **camera view**
2. Click the **scan/search icon** in the HUD

No `norma-vision-live` process is required. This finds dark blocks on a bright board (like your black rectangle on white foam).

Rebuild the viewer after UI changes:

```bash
cd software/station/bin/station && make client
```

Restart Station, then refresh `http://localhost:8889`.

## SmolVLA — arm control (repo VLA)

**SmolVLA** in `software/ai/smolvla_py/` is the repo’s Vision-Language-Action model. It outputs **joint motions**, not bounding boxes, so it does not replace the overlay. Use it to move the arm:

```bash
cd software/ai/smolvla_py
uv sync

uv run python scripts/run_policy.py \
  --checkpoint lerobot/smolvla_base \
  --task "pick up the black block" \
  --bus-serial YOUR_BUS_ID \
  --server localhost
```

Fine-tuned checkpoints work much better than the base model on your hardware. See [`software/ai/smolvla_py/README.md`](../../ai/smolvla_py/README.md).

Typical stack:

| Layer | Tool | Purpose |
|-------|------|---------|
| Where is the object? | Browser local vision (overlay) | Pixel box on camera |
| How should the arm move? | SmolVLA `run_policy.py` | Joint commands |

## Optional: Python live API (`norma-vision-live`)

Only needed if you want detections from Python (e.g. MCP) instead of the in-browser overlay:

```bash
uv run --project software/station/vision norma-vision-live \
  --backend contrast \
  --station-host localhost:8888
```

Use `--backend yoloe` only if you explicitly want Ultralytics YOLOE (heavy, often misses small blocks).

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `Vision error: Failed to fetch` | Rebuild viewer (`make client`) — overlay no longer needs port 8890 |
| Overlay on, no box | Object must be dark on a light surface; rebuild viewer |
| Arm doesn’t move | Use SmolVLA `run_policy.py`, not the overlay |
| YOLO finds nothing | Expected for small blocks — use local contrast instead |
