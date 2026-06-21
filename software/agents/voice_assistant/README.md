# NormaCore Voice Assistant

> Part of the [NormaCore Berlin Hackathon](../../../README.md) project.

Voice control for the ST3215 arm: speech → LLM tool calls → NormaCore MCP → Station.

**For demos, run the voice agent in [n8n](https://n8n.io)** — that is our recommended path. This folder is the **direct Codex/OpenAI API** implementation for local development and testing without n8n.

---

## Recommended: n8n voice workflow

Host the agent in **n8n** so prompts, wake-word handling, and MCP wiring are editable in a visual workflow:

1. **Trigger** — microphone input, webhook, or manual test
2. **STT** — OpenAI Whisper (or n8n speech node)
3. **LLM** — OpenAI / Codex with `norma-station` tool definitions
4. **Execute** — HTTP or MCP node → `go_home`, `transfer_object`, `say_hi`, etc.
5. **TTS** (optional) — short spoken confirmation

Point execution nodes at Station (`STATION_HOST=localhost:8888`) and the MCP server. See the [MCP README voice section](../../station/mcp/README.md#voice-agent-n8n--codex-api) and root [development stack](../../../README.md#hackathon-development-stack).

---

## Alternative: Python agent (Codex / OpenAI API direct)

Standalone script: microphone → Whisper → Chat Completions with tools → MCP stdio → `norma-station-mcp`.

### Prerequisites

1. **uv** — Python package manager
2. **Microphone** — connected to your computer
3. **OpenAI API key** — Whisper STT + Codex/GPT tool-calling (same OpenAI-compatible endpoint)
4. **NormaCore Station** — running with `--tcp` (port 8888)

### Installation

```bash
cp .env.example .env
# Set OPENAI_API_KEY in .env
```

### Usage

```bash
uv run agent.py
```

When it prints `READY!`, speak naturally:

- *"What is the current arm state?"*
- *"Go to home position"*
- *"Go right"* / *"Move up"* (`move_direction`)
- *"Move the object from position 9 to position 15"* (`transfer_object`)
- *"Say hi"* / *"Dance"*
- *"Hey Joe"* → `acknowledge`, then wait for command

### How it connects

```
agent.py  →  OpenAI API (Whisper + Codex/GPT tools)
          →  MCP stdio: uv run --project software/station/mcp python -m norma_station_mcp
          →  Station TCP localhost:8888
```

Use this when n8n is unavailable or you want a single-repo dev loop. Behavior should match the n8n workflow when both use the same system prompt and MCP tools.
