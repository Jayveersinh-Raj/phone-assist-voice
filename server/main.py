import os
import sys
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from .stt.factory import STTFactory
from .indic_trans import translate_en_hi  # existing function
from dotenv import load_dotenv
import openai
from .schemas.schema import TextRequest
import time

# Load env vars
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

# Initialize STT provider
stt_provider_type = os.getenv("STT_PROVIDER", "whisper")  # Default to whisper
stt_config = {}

if stt_provider_type == "deepgram":
    stt_config = {
        "api_key": os.getenv("DEEPGRAM_API_KEY"),
        "language": os.getenv("STT_LANGUAGE", "en"),
        "model": os.getenv("DEEPGRAM_MODEL", "nova-2")
    }
elif stt_provider_type == "whisper":
    stt_config = {
        "model": os.getenv("WHISPER_MODEL", "tiny"),
        "language": os.getenv("STT_LANGUAGE", "en"),
        "compute_type": os.getenv("WHISPER_COMPUTE_TYPE", "int8")
    }

try:
    stt_provider = STTFactory.create_provider(stt_provider_type, stt_config)
    print(f"[🔊] STT Provider initialized: {stt_provider_type}")
except Exception as e:
    print(f"[❌] Failed to initialize STT provider: {e}")
    # Fallback to whisper if available
    try:
        stt_provider = STTFactory.create_provider("whisper", {"model": "tiny"})
        print("[🔊] Fallback to Whisper STT provider")
    except Exception as fallback_error:
        print(f"[❌] Fallback failed: {fallback_error}")
        stt_provider = None

# Buffer and previous transcript
audio_buffer = []
prev_transcript = ""

print("[🔊] Ready to receive audio chunks.")
sys.stdout.flush()


# -----------------------------
# Helper: OpenAI translation
# -----------------------------
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
def translate_with_openai(text: str, target_language: str, source_language: str = None) -> str:
    if source_language:
        user_instr = (
            f"Translate the following text from {source_language} to {target_language}. "
            f"Output only the translated text, with no additional commentary, labels, or explanations.\n\n"
            f"{text}"
        )
    else:
        user_instr = (
            f"Translate the following text into {target_language}. "
            f"Output only the translated text, with no additional commentary, labels, or explanations.\n\n"
            f"{text}"
        )

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are a translation assistant. You must only output the translated result exactly, nothing else.",
            },
            {"role": "user", "content": user_instr},
        ],
        temperature=0.0,
        max_tokens=1000,
    )

    return response.choices[0].message.content.strip()

# -----------------------------
# Endpoints
# -----------------------------
@app.post("/stream/chunk")
async def stream_chunk(chunk: UploadFile = File(...)):
    global prev_transcript
    
    if stt_provider is None:
        return JSONResponse(content={"error": "STT provider not available"}, status_code=500)
    
    data = await chunk.read()
    pcm_data = np.frombuffer(data, dtype=np.int16)

    # Append new PCM data to buffer
    audio_buffer.extend(pcm_data)

    # Limit buffer to last 5 seconds (16kHz × 5s = 80,000 samples)
    if len(audio_buffer) > 80000:
        audio_buffer[:] = audio_buffer[-80000:]

    buffer_np = np.array(audio_buffer, dtype=np.int16)
    current_transcript = stt_provider.transcribe(buffer_np)

    if current_transcript.startswith(prev_transcript):
        new_part = current_transcript[len(prev_transcript):].strip()
    else:
        new_part = current_transcript

    prev_transcript = current_transcript

    if new_part:
        print(f"[🗣️ New] {new_part}")
        sys.stdout.flush()
        return {"transcript": new_part}

    return JSONResponse(content={"status": "waiting"}, status_code=200)


@app.post("/translate")
def translate(req: TextRequest):
    try:
        translated = translate_en_hi(req.text, req.src_lang, req.tgt_lang)
        return {"translated": translated}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/translate/openai")
def translate_openai(req: TextRequest):
    try:
        start = time.time()
        translated = translate_with_openai(req.text, req.tgt_lang, req.src_lang)
        end = time.time()
        print(f"Latency: {end - start}")
        print(f"Translated: {translated}")
        return {"translated": translated}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


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


@app.get("/stt/current")
def get_current_provider():
    """Get information about the current STT provider."""
    try:
        if stt_provider is None:
            return JSONResponse(content={"error": "No STT provider available"}, status_code=500)
        
        return {
            "provider_type": stt_provider_type,
            "config": stt_provider.get_config(),
            "supported_languages": stt_provider.get_supported_languages()
        }
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/stt/switch")
def switch_provider(provider_type: str, config: dict = None):
    """Switch to a different STT provider."""
    global stt_provider, stt_provider_type
    
    try:
        if provider_type not in STTFactory.get_available_providers():
            return JSONResponse(content={"error": f"Provider '{provider_type}' not available"}, status_code=400)
        
        new_provider = STTFactory.create_provider(provider_type, config or {})
        stt_provider = new_provider
        stt_provider_type = provider_type
        
        return {
            "message": f"Switched to {provider_type} provider",
            "provider_type": provider_type,
            "config": stt_provider.get_config()
        }
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/stt/language")
def set_stt_language(language: str):
    """Set the language for STT transcription."""
    try:
        if stt_provider is None:
            return JSONResponse(content={"error": "No STT provider available"}, status_code=500)
        
        stt_provider.set_language(language)
        return {
            "message": f"Language set to {language}",
            "language": language
        }
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
