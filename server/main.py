import os
import sys
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi import WebSocket, WebSocketDisconnect
from dotenv import load_dotenv, find_dotenv
import asyncio
from .stt import STTFactory

# Load env vars from repo-level .env (auto-discover), overriding existing vars if provided
load_dotenv(find_dotenv(), override=True)

# Also load server-local .env explicitly
_server_env = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(_server_env):
    load_dotenv(_server_env, override=True)
app = FastAPI()


@app.get("/health")
def health():
    return {"status": "ok"}


    


# -----------------------------
# STT Management Endpoints
# -----------------------------
@app.get("/stt/providers")
def get_available_providers():
    """Get list of available STT providers."""
    try:
        providers = STTFactory.get_available_providers()
        provider_info = {}
        for provider_type in providers:
            provider_info[provider_type] = STTFactory.get_provider_info(provider_type)
        return {"providers": provider_info}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


# -----------------------------
# WebSocket: Deepgram Streaming
# -----------------------------
@app.websocket("/ws/transcribe")
async def ws_transcribe(websocket: WebSocket):
    await websocket.accept()

    # Lazy import to avoid hard dependency when not used
    try:
        from deepgram import AsyncDeepgramClient
        from deepgram.core.events import EventType
        from deepgram.extensions.types.sockets import ListenV1ResultsEvent
    except Exception as e:
        await websocket.send_json({"error": f"Deepgram SDK not available: {e}"})
        await websocket.close()
        return

    api_key = os.getenv("DEEPGRAM_API_KEY")
    if not api_key:
        await websocket.send_json({"error": "DEEPGRAM_API_KEY not set"})
        await websocket.close()
        return

    try:
        dg = AsyncDeepgramClient(api_key=api_key)
        model = os.getenv("DEEPGRAM_MODEL", "nova-2")
        language = os.getenv("STT_LANGUAGE", "en")

        # Track final sentence assembly
        final_sentences = []
        
        # Open Deepgram listen v1 websocket
        async with dg.listen.v1.connect(
            model=model,
            language=language,
            encoding="linear16",
            sample_rate="16000",
            interim_results="true",
            punctuate="true",
            smart_format="true",
        ) as dg_socket:

            # Handle incoming messages
            async def handle_message(message):
                try:
                    if isinstance(message, ListenV1ResultsEvent):
                        alt = message.channel.alternatives[0] if message.channel.alternatives else None
                        text = alt.transcript if alt else ""
                        is_final = bool(message.is_final)
                        if not text:
                            return
                        if is_final:
                            print(f"[Final] {text}")
                            final_sentences.append(text)
                            await websocket.send_json({"final": text})
                        else:
                            print(f"[Partial] {text}")
                            await websocket.send_json({"partial": text})
                        sys.stdout.flush()
                except Exception:
                    pass

            dg_socket.on(EventType.MESSAGE, handle_message)

            # Start listening in background
            listen_task = asyncio.create_task(dg_socket.start_listening())

            # Receive audio frames from client and forward to Deepgram
            try:
                while True:
                    data = await websocket.receive_bytes()
                    if not data:
                        continue
                    await dg_socket.send_media(data)
            except WebSocketDisconnect:
                pass
            except Exception as e:
                try:
                    await websocket.send_json({"error": str(e)})
                except Exception:
                    pass
            finally:
                # Finalize and close
                try:
                    listen_task.cancel()
                except Exception:
                    pass
                try:
                    if final_sentences:
                        await websocket.send_json({"final_full": " ".join(final_sentences).strip()})
                except Exception:
                    pass
                try:
                    await websocket.close()
                except Exception:
                    pass
    except Exception as e:
        try:
            await websocket.send_json({"error": f"Failed to start Deepgram live: {e}"})
            await websocket.close()
        except Exception:
            pass


    
