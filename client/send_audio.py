import pyaudio
import requests
import time

CHUNK_DURATION_MS = 5000  # 5 seconds
SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2  # bytes (16-bit)
CHANNELS = 1
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION_MS / 1000)

SERVER_URL = "http://localhost:8000/stream/chunk"

def stream_from_microphone():
    p = pyaudio.PyAudio()

    stream = p.open(format=pyaudio.paInt16,
                    channels=CHANNELS,
                    rate=SAMPLE_RATE,
                    input=True,
                    frames_per_buffer=CHUNK_SIZE)

    print("[🎤] Recording from microphone... Press Ctrl+C to stop")

    try:
        while True:
            start = time.time()

            # Read raw audio data from mic
            audio_chunk = stream.read(CHUNK_SIZE, exception_on_overflow=False)

            files = {
                "chunk": ("chunk.pcm", audio_chunk, "application/octet-stream")
            }
            response = requests.post(SERVER_URL, files=files)

            if response.ok:
                json = response.json()
                if "transcript" in json:
                    print(f"[Transcript] {json['transcript']}")
                    end = time.time()
                    print(f"⏱️ Time taken: {end - start:.2f} sec")
            else:
                print(f"[⚠️] Server error: {response.status_code} - {response.text}")

    except KeyboardInterrupt:
        print("\n[🛑] Stopped by user.")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == "__main__":
    stream_from_microphone()





# import wave
# import requests
# import time

# CHUNK_DURATION_MS = 5000  # 5 seconds  # milliseconds
# SAMPLE_RATE = 16000
# SAMPLE_WIDTH = 2  # bytes
# CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION_MS / 1000) * SAMPLE_WIDTH
# SERVER_URL = "http://localhost:8000/stream/chunk"

# def send_audio_chunks(wav_path):
#     with wave.open(wav_path, 'rb') as wf:
#         assert wf.getframerate() == SAMPLE_RATE, "Audio must be 16kHz"
#         assert wf.getsampwidth() == SAMPLE_WIDTH, "Audio must be 16-bit"
#         assert wf.getnchannels() == 1, "Audio must be mono"

#         while True:
#             chunk = wf.readframes(CHUNK_SIZE // SAMPLE_WIDTH)
#             if not chunk:
#                 break

#             files = {"chunk": ("chunk.pcm", chunk, "application/octet-stream")}
#             start = time.time()
#             response = requests.post(SERVER_URL, files=files)

#             if response.ok:
#                 json = response.json()
#                 if "transcript" in json:
#                     print(f"[Transcript] {json['transcript']}")
#                     end = time.time()
#                     print(f"Time taken: {end-start}")
#             else:
#                 print(f"[⚠️] Error from server: {response.status_code} - {response.text}")

#             # Simulate real-time streaming
#             time.sleep(CHUNK_DURATION_MS / 1000.0)

# if __name__ == "__main__":
#     send_audio_chunks("test_audio/sample_english.wav")
