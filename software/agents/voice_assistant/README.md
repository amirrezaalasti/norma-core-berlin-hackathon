# NormaCore Voice Assistant

> Part of the [NormaCore Berlin Hackathon](../../../README.md) project.

A standalone Python agent that uses your microphone, OpenAI's Whisper STT, GPT-4o, and the NormaCore MCP server to control the ST3215 robot arm via voice.

## Prerequisites

1.  **uv**: Python package manager
2.  **Microphone**: A working microphone connected to your computer.
3.  **OpenAI API Key**: For Whisper STT and GPT-4o.

## Installation

1. Copy `.env.example` to `.env` and insert your OpenAI API key:
   ```bash
   cp .env.example .env
   # Edit .env with your favorite editor
   ```

2. Make sure the NormaCore Station is running in another terminal (with `--tcp` enabled, which is the default).

## Usage

Run the agent using `uv`:

```bash
uv run agent.py
```

It will install the dependencies, connect to the MCP server, load the tools, and start listening to your microphone.
When it prints `READY!`, you can start speaking.

Example commands:
- *"What is the current arm state?"*
- *"Go to home position"*
- *"Go right"* / *"Move up"* (uses `move_direction`)
- *"Open the gripper"*
- *"Pick up the object"*
- *"Go to square 9"* / *"Put it in square 5"*
- *"Say hi"* / *"Dance"*
