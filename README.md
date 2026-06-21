# NormaCore Berlin Hackathon

**Voice-controlled robotics with AI agents, computer vision, and MCP.**

This repository is our Berlin hackathon build on top of [NormaCore](https://normacore.dev) — an open robotics platform for real-time arm control, data collection, and deployment. We extended it with an MCP server, a voice assistant, vision-guided workspace calibration, and high-level pick/place tools so you can operate an ST3215 arm (ElRobot or SO-101) from Cursor, natural language, or speech.

---

## What we built

| Capability | Description |
|---|---|
| **MCP server** | 50+ tools exposed to AI assistants (Cursor, Claude, etc.) for joint motion, gripper control, pick/place, and board-square navigation |
| **Voice assistant** | Speech → MCP tool calls via **[n8n](https://n8n.io)** (recommended) or direct **Codex/OpenAI API** + local Python agent |
| **Vision stack** | Browser overlay (local contrast detection), optional Roboflow API, AprilTag mm calibration, and Python offline/live pipelines |
| **Board grid control** | 5×3 workspace grid with per-square pick and place (`go_to_square_8`, `place_at_square_3`, `transfer_object`, …) |
| **Pick & place** | Calibrated home pose, static pick pose, directional nudges, and lift/place sequences |
| **Station viewer UI** | Camera overlay, manual workspace calibration, and 5×3 grid on the live feed |
| **Lovable prototypes** | Future operator dashboard (board tap-to-move, transfer UI, voice status) — see [MCP README](software/station/mcp/README.md#ui-development-lovable--station-viewer) |
| **Personality** | `say_hi`, `acknowledge`, `dance`, and `gripper_wave` for demos and voice interaction |

---

## Demos

<table>
  <tr>
    <td align="center"><b>Say hi</b><br/><code>say_hi</code><br/><img src="videos/hi.gif" width="200" alt="Say hi demo"/></td>
    <td align="center"><b>Dance</b><br/><code>dance</code><br/><img src="videos/dance.gif" width="200" alt="Dance demo"/></td>
  </tr>
  <tr>
    <td align="center"><b>Pick up</b><br/><code>go_to_square</code> / pick<br/><img src="videos/pickup.gif" width="200" alt="Pick up demo"/></td>
    <td align="center"><b>Place</b><br/><code>place_at_square</code><br/><img src="videos/put.gif" width="200" alt="Place demo"/></td>
  </tr>
</table>

<p align="center">
  <sub>Loops are GIF previews from <code>videos/</code>. MP4 versions available for lighter embeds — see <a href="videos/README.md">videos/README.md</a>.</sub>
</p>

---

## Hackathon development stack

**Use [n8n](https://n8n.io) for voice and [Lovable](https://lovable.dev) for UI.** That was our intentional split during the hackathon: n8n orchestrates the voice agent and tool calls; Lovable prototypes the operator dashboard. The in-repo Python voice script and Station viewer changes are supporting implementations — not the primary demo path.

| Tool | Use for | Why |
|------|---------|-----|
| **[n8n](https://n8n.io)** | Voice agent workflow | Visual orchestration of wake word → STT → LLM → MCP/HTTP → robot; easy to tweak prompts and wire new tools without redeploying Python |
| **[Lovable](https://lovable.dev)** | Operator UI | Fast React dashboard for board grid, transfer flows, and live status — see [UI section](software/station/mcp/README.md#ui-development-lovable--station-viewer) |
| **Cursor + MCP** | Engineering / debugging | Direct access to all `norma-station` tools while calibrating and testing motion |
| **Station viewer** | Calibration + vision | Board corners, gripper tip, camera overlay, grid — ships in this repo |

### Architecture

```mermaid
flowchart LR
  subgraph inputs [Input]
    Voice[Microphone]
    Cursor[Cursor / MCP client]
  end

  subgraph agents [Agents]
    N8N[n8n Voice Workflow<br/>recommended]
    VA[Python agent<br/>Codex / OpenAI API]
    MCP[MCP Server<br/>norma-station-mcp]
  end

  subgraph station [NormaCore Station]
    TCP[TCP API :8888]
    UI[Web UI :8889]
    Cam[USB Camera]
  end

  subgraph ui_proto [UI — Lovable]
    Lovable[Lovable operator dashboard<br/>prototype]
  end

  subgraph robot [Hardware]
    Arm[ElRobot / SO-101]
    Board[Calibrated workspace]
  end

  Voice --> N8N
  Voice -.-> VA
  N8N --> MCP
  VA --> MCP
  Cursor --> MCP
  MCP --> TCP
  TCP --> Arm
  Cam --> UI
  Cam --> MCP
  Lovable -.-> MCP
  Board --> Arm
```

---

## Quick start

### Prerequisites

- macOS (Apple Silicon tested) or Linux
- ST3215 robot arm connected via USB (ElRobot 7+1 DoF or SO-101 5+1 DoF)
- [uv](https://docs.astral.sh/uv/) — `curl -LsSf https://astral.sh/uv/install.sh | sh`
- Generated protobufs (from repo root):

```bash
make protobuf
```

### 1. Start NormaCore Station

Station bridges USB motors to a TCP API on port **8888** and serves the web UI on **8889**.

**Prebuilt binary (recommended):**

```bash
mkdir -p .tmp/station && cd .tmp/station

curl -L -o station-macos-arm64.zip \
  "https://github.com/norma-core/norma-core/releases/download/v0.1.0-beta.8/station-macos-arm64.zip"

unzip -o station-macos-arm64.zip
cp ../../software/station/bin/station/station.yaml .

RUST_LOG=info ./station --tcp --web --config station.yaml
```

Open **http://localhost:8889** to verify live motor data and the camera overlay.

Full setup options (build from source, desktop app, troubleshooting): [`software/station/mcp/README.md`](software/station/mcp/README.md)

### 2. Control from Cursor (MCP)

The repo ships `.cursor/mcp.json`. After station is running:

1. Reload MCP servers in Cursor (Settings → MCP → refresh)
2. Confirm **norma-station** appears with tools like `get_arm_state`, `go_home`, `pick_object`

Example prompts:

- *"What is the arm state?"*
- *"Go home and open the gripper"*
- *"Pick up the object and lift"*
- *"Go to square 9"*
- *"Say hi"*

### 3. Voice control

We run voice two ways. **Prefer n8n** for demos and iteration; use the **local Python agent** when you want a single-repo script calling the Codex/OpenAI API directly.

#### Option A — n8n voice workflow (recommended)

The production demo path hosts the voice agent in **[n8n](https://n8n.io)**:

1. **Trigger** — microphone / webhook / scheduled listen (wake word: *"hey joe"*)
2. **STT** — Whisper or n8n speech node → transcript
3. **LLM** — OpenAI / Codex with MCP tool definitions (`go_home`, `transfer_object`, `say_hi`, …)
4. **MCP / HTTP** — call `norma-station` tools against Station on `localhost:8888`
5. **TTS** (optional) — short spoken confirmation

Benefits: change system prompts and tool wiring in the n8n canvas without touching Python; easy to add logging, retries, and Slack/Discord notifications.

Import or recreate the workflow in your n8n instance, point MCP/HTTP nodes at the running Station + MCP server, and ensure `STATION_HOST=localhost:8888`.

#### Option B — Python agent + Codex/OpenAI API (direct)

The repo includes a standalone script that talks to the **Codex/OpenAI API** directly (no n8n):

```bash
cd software/agents/voice_assistant
cp .env.example .env   # add OPENAI_API_KEY (Codex-compatible OpenAI endpoint)
uv run agent.py
```

Flow: microphone → Whisper STT → Chat Completions + tool calls → MCP stdio → `norma-station-mcp` → Station.

When you see `READY!`, speak naturally:

- *"Go right"*, *"Move up a bit"*
- *"Pick up the object"*
- *"Move the object from position 9 to position 15"* → `transfer_object`
- *"Hey Joe, can you hear me?"* → `acknowledge`

Details: [`software/agents/voice_assistant/README.md`](software/agents/voice_assistant/README.md)

### 4. Vision & workspace calibration (optional)

Copy `.env.example` to the repo root as `.env` and configure Roboflow if using cloud detection:

```bash
cp .env.example .env
```

In the station viewer: switch to **camera view** → click the **scan icon** for live object detection. Calibrate the workspace board for grid pick/place.

Details: [`software/station/vision/README.md`](software/station/vision/README.md)

---

## MCP tool reference

### High-level (preferred)

| Tool | Purpose |
|---|---|
| `get_arm_state` | Read joints, gripper, and detected arm type |
| `go_home` | Return to saved home pose in `.norma/home_pose.json` |
| `pick_object` / `lift_object` / `place_object` | Static pick/place sequence |
| `go_to_square` / `go_to_square_N` | Move to board square 1–15 and grasp |
| `place_at_square` / `place_at_square_N` | Place held object at a square |
| `transfer_object` | Pick at `from_square` and place at `to_square` in one call |
| `move_direction` | Calibrated up / down / left / right nudge |
| `say_hi` / `acknowledge` / `dance` | Demo gestures |
| `detect_workspace_objects` | Vision offset from gripper (optional) |

### Low-level

| Tool | Purpose |
|---|---|
| `move_joint` / `move_arm_pose` | Joint-space motion (0.0–1.0 per motor) |
| `open_gripper` / `close_gripper` / `set_gripper` | Gripper control |
| `enable_arm_torque` / `disable_arm_torque` | Motor power |
| `advanced_*` | Raw motor bus access for debugging |

Joint IDs match motor IDs: **SO-101** joints 1–5 + gripper 6; **ElRobot** joints 1–7 + gripper 8. Positions are normalized within each motor's calibrated range, not Cartesian XYZ.

---

## Repository layout

```
norma-core-berlin-hackathon/
├── software/
│   ├── station/
│   │   ├── mcp/                  # MCP server (hackathon core)
│   │   ├── vision/               # Detection, workspace, AprilTags
│   │   ├── clients/station-viewer/  # Browser UI + local vision overlay
│   │   └── bin/station/          # NormaCore Station (Rust)
│   ├── agents/
│   │   └── voice_assistant/      # Direct Codex/OpenAI API agent (n8n is recommended for demos)
│   └── ai/smolvla_py/            # SmolVLA policy training & inference
├── hardware/
│   ├── elrobot/                  # 7+1 DoF arm (3D-printable)
│   └── pgripper/                 # Parallel jaw gripper
├── shared/
│   ├── gremlin_go/               # Protobuf SDK (Go)
│   └── gremlin_py/               # Protobuf SDK (Python)
├── .norma/                       # Local calibration (home pose, grid, nudges)
├── videos/                       # Demo clips (GIF / MP4 for README, MOV sources)
├── .cursor/mcp.json              # Cursor MCP configuration
└── .env.example                  # Vision / Roboflow environment template
```

---

## Calibration files

Local robot state lives under `.norma/` (gitignored secrets in `.env`):

| File | Purpose |
|---|---|
| `home_pose.json` | Arm rest pose — set via `save_home_pose` |
| `pick_calibration.json` | Board workspace + per-square joint targets |
| `manual_workspace.json` | Viewer workspace corners for vision overlay |
| `direction_nudge.json` | Teleop-calibrated directional joint deltas |

---

## NormaCore platform

This hackathon fork builds on the full NormaCore toolkit:

| Project | Path | Description |
|---|---|---|
| **ElRobot** | [`hardware/elrobot/`](hardware/elrobot/) | Fully 3D-printed 7+1 DoF arm for imitation learning |
| **Parallel Jaw Gripper** | [`hardware/pgripper/`](hardware/pgripper/) | Modular gripper for the SO-101 arm |
| **Station** | [`software/station/bin/station/`](software/station/bin/station/) | Real-time robotics platform — data collection, inference, control |
| **SmolVLA** | [`software/ai/smolvla_py/`](software/ai/smolvla_py/) | Train + deploy a [SmolVLA](https://huggingface.co/docs/lerobot/smolvla) policy |
| **Gremlin** | [`shared/gremlin_go/`](shared/gremlin_go/) · [`shared/gremlin_py/`](shared/gremlin_py/) | High-performance Protobuf SDK for Go and Python |

**Website:** [normacore.dev](https://normacore.dev)

**Community:**
- [Discord](https://discord.gg/Z4Ytw3QfHP)
- [GitHub](https://github.com/norma-core/norma-core)
- [X/Twitter](https://x.com/norma_core_dev)
- [YouTube](https://www.youtube.com/@normacoredev)

---

## Troubleshooting

| Problem | Fix |
|---|---|
| Nothing on port 8888 | Start station with `--tcp` |
| MCP tools fail | Station must be running first; reload MCP in Cursor |
| `Missing generated protobufs` | Run `make protobuf` from repo root |
| Arm won't move | Call `enable_arm_torque` before motion commands |
| Voice assistant errors | n8n: check workflow credentials and MCP/HTTP node URL; direct agent: set `OPENAI_API_KEY` in `software/agents/voice_assistant/.env` |
| Vision shows pixels not mm | Calibrate with AprilTags or camera intrinsics/extrinsics |

See also: [MCP setup guide](software/station/mcp/README.md) · [Vision guide](software/station/vision/README.md)

---

## License & attribution

Built at the **NormaCore Berlin Hackathon** on the open-source [NormaCore](https://github.com/norma-core/norma-core) platform.
