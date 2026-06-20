# NormaCore Station — Robot Connection & MCP Setup

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
  "has_latest_inference": true,
  "bus_count": 1
}
```

---

## 4. Enable MCP in Cursor

The project includes `.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "norma-station": {
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
        "STATION_HOST": "localhost:8888"
      }
    }
  }
}
```

1. Ensure station is running (step 2).
2. Reload MCP servers in Cursor (Settings → MCP → refresh, or restart Cursor).
3. The **norma-station** server should appear with tools like `get_arm_state`, `move_joint`, `open_gripper`, etc.

If the robot runs on another machine, change `STATION_HOST` to e.g. `"192.168.1.100:8888"`.

---

## 5. Control the robot from Cursor

Recommended order:

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

## 6. Stop station

In the terminal where station is running, press **Ctrl+C**.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `Nothing listening on 8888` | Start station with `--tcp` (step 2). |
| No `/dev/cu.usb*` device | Replug USB cable; try a different port/cable. |
| `No st3215/inference frames` | Confirm `st3215.enabled: true` in `station.yaml`; check station logs for port errors. |
| MCP tools fail to connect | Station must be running first; reload MCP in Cursor. |
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
