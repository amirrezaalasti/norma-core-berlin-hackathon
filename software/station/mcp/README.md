# NormaCore Station — Robot Connection & MCP Setup

> Part of the [NormaCore Berlin Hackathon](../../../README.md) project. This guide covers station startup, MCP configuration, and robot control from Cursor.

Step-by-step guide to connect an ST3215 robot arm to your laptop, verify it works, and control it from Cursor via MCP.

Tested on **macOS Apple Silicon** with an **ElRobot** arm (7 joints + gripper on motor 8).

---

## Prerequisites

- Robot connected via USB
- [uv](https://docs.astral.sh/uv/) installed (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- Generated protobufs (from repo root):

```bash
make protobuf
```

---

## 1. Connect the robot

Plug the robot into your laptop over USB. Confirm macOS sees the serial port:

```bash
ls /dev/cu.usb*
```

You should see something like `/dev/cu.usbmodem5B3E0897161`.

---

## 2. Start NormaCore Station

Station bridges the USB motors to a TCP API on port **8888**. The MCP server talks to that port.

### Option A — Prebuilt binary (recommended)

Download the macOS release binary:

```bash
mkdir -p .tmp/station && cd .tmp/station

curl -L -o station-macos-arm64.zip \
  "https://github.com/norma-core/norma-core/releases/download/v0.1.0-beta.8/station-macos-arm64.zip"

unzip -o station-macos-arm64.zip
cp ../../software/station/bin/station/station.yaml .
```

Start station with TCP and web UI:

```bash
RUST_LOG=info ./station --tcp --web --config station.yaml
```

Leave this terminal running.

### Option B — Build from source

```bash
cd software/station/bin/station
make client          # builds web UI assets (required)
cargo build --release
./../../../../target/release/station --tcp --web
```

### Option C — Desktop app

Install [station-macos-arm64.dmg](https://github.com/norma-core/norma-core/releases/tag/v0.1.0-beta.8) and launch **NormaCore Station**. It starts the TCP server automatically.

---

## 3. Verify the connection

### Web UI

Open **http://localhost:8889** in your browser. You should see the arm with live motor data.

### Station logs

In the station terminal, look for:

```
Successfully opened ST3215 port: /dev/tty.usbmodem...
Detected new ST3215 motor ID on port: ... 1
Detected new ST3215 motor ID on port: ... 2
...
NormFS server listening on 0.0.0.0:8888
```

For an ElRobot you should see **8 motors** (joints 1–7, gripper 8). For an SO-101 you should see **6 motors** (joints 1–5, gripper 6).

### Python connection test

From the repo root:

```bash
uv run --project software/station/mcp python -c "
import asyncio, json
from norma_station_mcp.session import StationSession

async def main():
    s = StationSession('localhost:8888')
    await s.ensure_connected()
    await s.wait_for_inference(timeout_s=15.0)
    info = s.connection_info()
    info['bus_count'] = len(s.list_buses())
    state = s.get_arm_state()
    print(json.dumps(info, indent=2))
    print(json.dumps({
        'arm_type': state['arm_type'],
        'arm_label': state['arm_label'],
        'bus_serial': state['bus_serial'],
        'joint_count': len(state['joints']),
        'joint_ids': [j['motor_id'] for j in state['joints']],
        'gripper_motor_id': state['gripper_motor_id'],
    }, indent=2))

asyncio.run(main())
"
```

Expected output:

```json
{
  "host": "localhost:8888",
  "connected": true,
  "setup_done": true,
  "frame_count": 1,
  "has_latest_inference": true,
  "last_error": null,
  "bus_count": 1
}
```

---

## 4. Enable MCP (Claude Code terminal + Cursor)

We primarily used **Claude Code in the terminal** to drive the MCP server during the hackathon. **Cursor** uses the same `norma-station` tools via a parallel config.

### Claude Code (terminal — recommended)

The repo includes [`.mcp.json`](../../../.mcp.json) at the project root:

```json
{
  "mcpServers": {
    "norma-station": {
      "type": "stdio",
      "command": "uv",
      "args": [
        "run",
        "--project",
        "software/station/mcp",
        "python",
        "-m",
        "norma_station_mcp"
      ],
      "env": {
        "STATION_HOST": "localhost:8888",
        "NORMA_MOTOR_SPEED_SCALE": "0.5"
      }
    }
  }
}
```

1. Ensure station is running (step 2).
2. From the repo root: `claude`
3. Type `/mcp` and approve **norma-station** on first use.
4. Ask Claude to control the arm (`go_home`, `say_hi`, `transfer_object`, …).

Or add via CLI:

```bash
claude mcp add --scope project norma-station -- \
  uv run --project software/station/mcp python -m norma_station_mcp
```

Docs: [Claude Code MCP](https://code.claude.com/docs/en/mcp)

### Cursor (IDE)

The project also includes [`.cursor/mcp.json`](../../../.cursor/mcp.json) with the same server definition as [`.mcp.json`](../../../.mcp.json).

1. Ensure station is running (step 2).
2. Open this repo as the Cursor workspace (MCP runs with the workspace as cwd).
3. **Settings → MCP → refresh** (or restart Cursor) after pulling MCP changes — Cursor caches tool lists; a refresh is required to pick up new tools like `say_hi`, `dance`, `transfer_object`, and `go_to_square_N`.
4. Confirm **norma-station** is connected (green). You should see **67 tools**, including:
   - **Gestures:** `say_hi`, `acknowledge`, `gripper_wave`, `dance`
   - **Board:** `go_to_square`, `go_to_square_1`…`15`, `place_at_square`, `place_at_square_1`…`15`, `transfer_object`, `list_square_poses`
   - **Pick/place:** `pick_object`, `lift_object`, `place_object`, `go_home`
   - **Motion:** `move_direction`, `move_joint`, `move_arm_pose`, `open_gripper`, `close_gripper`
   - **State:** `get_arm_state`, `station_connection_status`, `get_home_pose`

If the robot runs on another machine, change `STATION_HOST` to e.g. `"192.168.1.100:8888"` in both config files.

Verify the local tool surface (optional):

```bash
uv run --project software/station/mcp python -c "
import asyncio
from norma_station_mcp.server import mcp
print(len(asyncio.run(mcp.get_tools())), 'tools')
"
```

---

## 5. Control the robot via MCP

### Pick / place workflow (recommended)

Use these high-level MCP tools in order:

1. **`get_arm_state`** — read current joint and gripper positions
2. **`pick_object`** — home → open gripper → move to fixed pick pose → close gripper
3. **`lift_object`** — move to home while holding the object (gripper stays closed)
4. **`place_object`** — move to pick pose → open gripper → return home
5. **`go_home`** — move to saved home pose (set `open_gripper=false` if holding an object)

Poses: home from `.norma/home_pose.json`; pick/placement joints are **static** (hardcoded in `pick_control.py`, not vision-derived).

### Directional nudges (up / down / left / right)

Use **`move_direction`** for commands like “go right” or “move up a bit”:

- **`move_direction`** — nudges toward motor-range endpoints from the **current** pose (not fixed joint coordinates)
- `direction`: `up`, `down`, `left`, or `right`
- ElRobot: **left/right** → motor 1 (1176 ← home → 2920); **up/down** → motors 2 and 3
- `amount`: `1.0` ≈ 10% of each motor span (default); `2.0` = double nudge
- Optional override: `.norma/direction_nudge.json` (built-in ElRobot endpoints when missing)

Do **not** guess single-joint moves for directions — use `move_direction` instead.

### Board grid (squares 1–15)

The workspace is a **5×3 grid** (configurable via `NORMA_BOARD_GRID_COLS` / `NORMA_BOARD_GRID_ROWS`). Use square tools for chess-like pick/place:

1. **`transfer_object`** — pick at `from_square` and place at `to_square` in one call (preferred for voice: *"move from 9 to 15"*)
2. **`go_to_square`** or **`go_to_square_N`** — move to square N and grasp (partial gripper close)
3. **`place_at_square`** or **`place_at_square_N`** — place held object at square N (only after a pick)
4. **`list_square_poses`** / **`get_square_pose`** — inspect per-square joint targets

Calibrate the board in the station viewer first; joint targets are stored in `.norma/pick_calibration.json`.

### Fun gestures

- **`say_hi`** — fully open/close gripper twice (voice: *"say hi"*)
- **`acknowledge`** — quick head nod when called (voice: *"hey Joe"*)
- **`dance`** — energetic arm sway with gripper flaps
- **`gripper_wave`** — rapid gripper open/close

### Low-level motion

1. **`get_arm_state`** — read current joint and gripper positions (start here).
2. **`enable_arm_torque`** — power motors so the arm holds position.
3. **`move_joint`** / **`move_arm_pose`** — move joints (values 0.0–1.0, normalized per motor).
4. **`open_gripper`** / **`close_gripper`** / **`set_gripper`** — gripper control.
5. **`disable_arm_torque`** — release the arm (use with care).

### Joint reference

| Arm | Joints | Gripper |
|-----|--------|---------|
| SO-101 | motors 1–5 | motor 6 |
| ElRobot | motors 1–7 | motor 8 |

Positions are **normalized 0.0–1.0** within each motor's calibrated range, not Cartesian XYZ.

---

## UI development (Lovable + Station viewer)

**Use [Lovable](https://lovable.dev) for operator UI.** We intentionally prototyped the dashboard there rather than extending the Rust-embedded Station viewer for every demo screen. Station viewer changes in this repo cover calibration and vision; Lovable covers tap-to-move and status UX.

We used two UI tracks during the hackathon: **in-repo Station viewer** changes for calibration and vision, and **Lovable** for rapid prototyping of a future operator-facing dashboard.

### Station viewer (shipped in this repo)

The NormaCore Station web UI at **http://localhost:8889** (`software/station/clients/station-viewer/`) was extended for board pick/place workflows:

| Feature | Location | Purpose |
|---------|----------|---------|
| **Local contrast vision** | Camera HUD → scan icon | Detect dark blocks on the board in-browser; no cloud API required |
| **Manual workspace calibration** | Camera view click workflow | Set 4 board corners (TL/TR/BR/BL) + gripper tip; saved to `.norma/manual_workspace.json` |
| **5×3 board grid overlay** | `workspace-grid.ts` + vision overlay | Draws grid lines and square IDs (1–15) on the live camera feed |
| **Square-aware detections** | `enrichDetectionWithSquare` | Labels each detection with `square_id` and offset from cell center |
| **Roboflow path** | Vision API + env config | Optional cloud detection when `ROBOFLOW_API_KEY` is set |

Rebuild the embedded viewer after UI changes:

```bash
cd software/station/bin/station && make client
```

Details: [`software/station/vision/README.md`](../vision/README.md)

### Lovable prototypes (future operator UI)

We used **Lovable** to iterate quickly on a separate **operator dashboard** concept — a simpler surface than the full Station viewer, aimed at demos and day-to-day board control. These screens were designed against the same mental model as the MCP tools (`go_home`, `go_to_square_N`, `place_at_square_N`, `transfer_object`).

**Prototyped flows:**

- **Board control panel** — visual 5×3 grid; tap a square to pick or place; highlight current arm target
- **Object transfer** — “from square → to square” picker wired to the `transfer_object` sequence
- **Live status strip** — connection state, gripper open/closed, last motion result
- **Camera + grid preview** — side-by-side arm status and overlaid workspace (mirrors Station viewer grid)
- **Calibration wizard** — guided corner + gripper-tip setup for non-engineers
- **Voice assistant panel** — transcript, last MCP tool call, and quick gesture buttons (`say_hi`, `acknowledge`)

**Not yet integrated:** the Lovable app is a design prototype; production wiring would call the Station TCP API (port 8888) or MCP over HTTP/WebSocket. Until then, use **n8n voice**, Cursor MCP, or the Station viewer for real motion.

---

## Voice agent (n8n + Codex API)

Voice control follows the same MCP tools as Cursor. **Run the agent in n8n for demos**; use the in-repo Python script when you need direct Codex/OpenAI API access.

### n8n workflow (recommended)

Host the voice agent in **[n8n](https://n8n.io)** — this is what we used for live hackathon demos:

| Step | n8n node (typical) | Output |
|------|-------------------|--------|
| 1 | Trigger (webhook, mic, or manual) | Raw audio or text |
| 2 | OpenAI Whisper / speech-to-text | Transcript |
| 3 | OpenAI / Codex chat with tools | Tool call JSON (`go_home`, `transfer_object`, …) |
| 4 | HTTP Request or MCP node | Execute against `norma-station` |
| 5 | OpenAI TTS (optional) | Spoken reply |

**Why n8n:** prompt and tool routing live in one visual workflow; non-developers can adjust wake-word behavior, add guards (*"never place without pick"*), and log every command without redeploying code.

Requirements: Station running with `--tcp`, MCP server reachable from n8n (stdio wrapper, HTTP bridge, or subprocess `uv run --project software/station/mcp python -m norma_station_mcp`).

### Direct Codex / OpenAI API (alternative)

`software/agents/voice_assistant/agent.py` implements the same loop locally:

```
Microphone → Whisper (OpenAI API) → Codex/GPT tool calls → MCP stdio → norma-station-mcp → Station :8888
```

```bash
cd software/agents/voice_assistant
cp .env.example .env   # OPENAI_API_KEY
uv run agent.py
```

Use this path for development, CI, or when n8n is not available. Both approaches call the **same MCP tool surface** documented in section 5.

Details: [`software/agents/voice_assistant/README.md`](../../agents/voice_assistant/README.md)

---

## 6. Stop station

In the terminal where station is running, press **Ctrl+C**.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `Nothing listening on 8888` | Start station with `--tcp` (step 2). |
| No `/dev/cu.usb*` device | Replug USB cable; try a different port/cable. |
| `No st3215/inference frames` | Confirm `st3215.enabled: true` in `station.yaml`; check station logs for port errors. |
| MCP tools fail to connect | Station must be running first; Claude Code: `/mcp` and approve `norma-station`; Cursor: reload MCP |
| `Missing generated protobufs` | Run `make protobuf` from repo root. |
| Arm detected but won't move | Run `enable_arm_torque` before sending move commands. |
| Build fails on `Asset::get` | Run `make client` in `software/station/bin/station` before `cargo build`. |

---

## Quick reference

```bash
# Repo root
make protobuf

# Start station (prebuilt)
cd .tmp/station && RUST_LOG=info ./station --tcp --web --config station.yaml

# Test connection
uv run --project software/station/mcp python -c "..."   # see step 3

# Web UI
open http://localhost:8889
```
