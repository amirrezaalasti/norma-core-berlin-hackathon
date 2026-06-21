import asyncio
import json
import os
import re
from typing import List, Dict, Any

from openai import AsyncOpenAI
import speech_recognition as sr
from dotenv import load_dotenv

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Load environment variables from .env file
load_dotenv()

openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

async def run_agent():
    print("Starting voice assistant...")
    
    # Check if API key is present
    if not os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") == "your-openai-api-key-here":
        print("ERROR: Please set your OPENAI_API_KEY in the .env file.")
        return

    # Configure the MCP server connection
    mcp_script = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "station", "mcp"))
    server_params = StdioServerParameters(
        command="uv",
        args=["run", "--project", mcp_script, "python", "-m", "norma_station_mcp"],
        env=dict(os.environ, STATION_HOST=os.getenv("STATION_HOST", "localhost:8888"))
    )

    print(f"Connecting to MCP server at {mcp_script}...")
    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                print("Connected to MCP server.")
                
                # Fetch tools from MCP server
                mcp_tools = await session.list_tools()
                
                # Convert to OpenAI tool format
                openai_tools = []
                for t in mcp_tools.tools:
                    openai_tools.append({
                        "type": "function",
                        "function": {
                            "name": t.name,
                            "description": t.description,
                            "parameters": t.inputSchema
                        }
                    })
                
                print(f"Loaded {len(openai_tools)} tools.")

                messages = [
                    {
                        "role": "system",
                        "content": (
                            "You are a helpful voice assistant controlling an ST3215 robot arm via MCP tools. "
                            "Execute the user's voice commands efficiently. Use tools when needed. "
                            "For directional motion (up, down, left, right, go home), prefer move_direction and "
                            "go_home instead of guessing individual joint values. "
                            "Use amount=2.0 in move_direction when the user asks for a bigger or more visible move. "
                            "Keep your verbal responses extremely short and concise, since this is a voice conversation."
                        )
                    }
                ]

                recognizer = sr.Recognizer()
                mic = sr.Microphone()
                
                import queue
                import time
                
                audio_queue = queue.Queue()
                def audio_callback(recognizer, audio):
                    audio_queue.put(audio)

                print("Adjusting for ambient noise... please wait.")
                with mic as source:
                    recognizer.adjust_for_ambient_noise(source, duration=2)
                    
                print("\n" + "="*50)
                print("READY! Speak into your microphone to control the robot.")
                print("Your speech will be transcribed in natural phrases.")
                print("Press Ctrl+C to exit.")
                print("="*50 + "\n")
                
                # Use natural pause detection instead of aggressive 2-second chunking
                # This prevents cutting words in half and makes transcription MUCH more stable.
                recognizer.pause_threshold = 0.6
                recognizer.non_speaking_duration = 0.4
                stop_listening = recognizer.listen_in_background(mic, audio_callback)
                
                current_phrase = []
                is_recording = False
                
                print("Listening for 'hey joe'...")
                
                while True:
                    try:
                        # Non-blocking get
                        audio = audio_queue.get_nowait()
                        
                        # Process the chunk
                        temp_file = "temp_audio.wav"
                        with open(temp_file, "wb") as f:
                            f.write(audio.get_wav_data())
                        
                        try:
                            with open(temp_file, "rb") as f:
                                transcript_response = await openai_client.audio.transcriptions.create(
                                    model="whisper-1",
                                    file=f,
                                    prompt="Hey joe. pick the black box. command end.",
                                    temperature=0.0
                                )
                            chunk_text = transcript_response.text.strip()
                            
                            # Whisper often hallucinates on silence. Filter out common ones:
                            hallucinations = ["Thank you.", "Thank you", "Thanks for watching", "I'm sure we're going to have a lot of fun with it.", "You", "Yeah.", "I mean, what can you say?"]
                            is_hallucination = any(chunk_text.lower().replace(".", "").replace(",", "") == h.lower().replace(".", "").replace(",", "") for h in hallucinations)
                            
                            if not chunk_text or is_hallucination:
                                continue
                                
                            chunk_lower = chunk_text.lower()
                            
                            start_match = re.search(r'hey\s*[,.]*\s*joe', chunk_lower)
                            if start_match:
                                is_recording = True
                                current_phrase = []
                                print("\n\033[92m[STARTED RECORDING COMMAND]\033[0m Speak your command...")
                                
                                # Extract anything said AFTER 'hey joe'
                                start_idx = start_match.end()
                                
                                if len(chunk_text) > start_idx:
                                    after_text = chunk_text[start_idx:].strip()
                                    # Strip leading punctuation that might be left over
                                    after_text = re.sub(r'^[.,!?\s]+', '', after_text)
                                    if after_text:
                                        current_phrase.append(after_text)
                                        print(f"\r\033[KUser: {' '.join(current_phrase)}", end="", flush=True)
                                continue
                                
                            # Check for various natural ways to end the command
                            end_match = re.search(r'(?:command|joe)\s*[,.]*\s*(?:end|stop)|(?:end|stop)\s*[,.]*\s*(?:command|joe)', chunk_lower)
                            if end_match and is_recording:
                                is_recording = False
                                
                                # Extract anything said BEFORE 'command end'
                                end_idx = end_match.start()
                                
                                if end_idx > 0:
                                    before_text = chunk_text[:end_idx].strip()
                                    if before_text:
                                        current_phrase.append(before_text)
                                
                                print("\n\033[93m[COMMAND RECORDED]\033[0m Sending to AI...")
                                
                                full_text = " ".join(current_phrase)
                                current_phrase = []
                                
                                if not full_text.strip():
                                    print("Empty command. Listening for 'hey joe'...")
                                    continue
                                
                                messages.append({"role": "user", "content": full_text})
                                
                                while True:
                                    try:
                                        response = await openai_client.chat.completions.create(
                                            model="gpt-4o",
                                            messages=messages,
                                            tools=openai_tools,
                                            tool_choice="auto"
                                        )
                                    except Exception as e:
                                        print(f"LLM error: {e}")
                                        break
                                    
                                    choice = response.choices[0]
                                    message = choice.message
                                    
                                    if message.content:
                                        print(f"\nAssistant: {message.content}\n")
                                    
                                    messages.append(message.model_dump(exclude_none=True))
                                    
                                    if message.tool_calls:
                                        for tool_call in message.tool_calls:
                                            name = tool_call.function.name
                                            args = json.loads(tool_call.function.arguments)
                                            print(f"-> Executing: {name}({args})")
                                            
                                            try:
                                                result = await session.call_tool(name, arguments=args)
                                                result_text = "\n".join(
                                                    [c.text for c in result.content if hasattr(c, 'text')]
                                                )
                                                if not result_text:
                                                    result_text = str(result)
                                                print(f"<- Result: {result_text[:200]}...")
                                            except Exception as e:
                                                result_text = f"Error executing tool: {e}"
                                                print(f"<- {result_text}")
                                                
                                            messages.append({
                                                "role": "tool",
                                                "tool_call_id": tool_call.id,
                                                "name": name,
                                                "content": result_text
                                            })
                                    else:
                                        # GPT didn't call any more tools, we're done processing this voice command
                                        print("\nListening for 'hey joe'...")
                                        break
                                        
                            elif is_recording:
                                current_phrase.append(chunk_text)
                                print(f"\r\033[KUser: {' '.join(current_phrase)}", end="", flush=True)
                                
                        except Exception as e:
                            pass # Silently ignore transcription errors on chunks to avoid clutter

                    except queue.Empty:
                        # Small sleep to prevent busy waiting
                        await asyncio.sleep(0.1)

    except Exception as e:
        print(f"MCP Connection Error: {e}")

if __name__ == "__main__":
    try:
        import sys
        import warnings
        warnings.filterwarnings("ignore")
        asyncio.run(run_agent())
    except KeyboardInterrupt:
        print("\nExiting...")
        if os.path.exists("temp_audio.wav"):
            try:
                os.remove("temp_audio.wav")
            except:
                pass
