import wave
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse, Response
from piper import PiperVoice
from pathlib import Path
import io
import subprocess
import tempfile
import os
from pydantic import BaseModel
import re

app = FastAPI()

PIPER_MODEL = Path("models/piper/en_US-hfc_female-medium.onnx")
WHISPER_CLI = Path("whisper.cpp/build/bin/whisper-cli")
WHISPER_MODEL = Path("whisper.cpp/models/ggml-base.en.bin")

voice = PiperVoice.load(PIPER_MODEL)

class TTSRequest(BaseModel):
    text: str


# @app.post("/tts")
# async def tts(req: TTSRequest):
#     text = req.text.strip()
#     if not text:
#         raise HTTPException(400, "Text required")

#     print(f"[TTS] Received text (len={len(text)}): '{text}'")

#     wav_io = io.BytesIO()

#     try:
#         with wave.open(wav_io, 'wb') as wav_file:
#             voice.synthesize_wav(text, wav_file)

#         wav_io.seek(0)
#         audio_bytes = wav_io.read()

#         print(f"[TTS] Synthesized {len(audio_bytes)} bytes")
        
#         wav_io.seek(0)

#         return Response(
#             content=audio_bytes,
#             media_type="audio/wav",
#             headers={"Content-Disposition": "attachment; filename=speech.wav"}
#         )

#     except Exception as e:
#         print(f"[TTS] Synthesis failed: {type(e).__name__}: {e}")
#         import traceback
#         traceback.print_exc()
#         raise HTTPException(500, f"TTS synthesis failed: {str(e)}")

# -------------------------------------------------------------------
@app.post("/tts")
async def tts(req: TTSRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(400, "Text required")

    # Optional: minor cleaning to help phonemization
    text = re.sub(r'\s+', ' ', text).strip()

    def generate_pcm_chunks():
        for chunk in voice.synthesize(text):  # ← this is the generator!
            # Get raw 16-bit PCM bytes directly
            pcm_bytes = chunk.audio_int16_bytes
            if pcm_bytes:
                yield pcm_bytes

    return StreamingResponse(
        generate_pcm_chunks(),
        media_type="audio/pcm;rate=22050",  # Hint (client can ignore)
        headers={
            "Content-Type": "audio/pcm",
            "X-Sample-Rate": str(voice.config.sample_rate),  # Optional metadata
        }
    )
# -------------------------------------------------------------------

@app.post("/stt")
async def stt(file: UploadFile = File(...)):
    if not file.filename.endswith(".wav"):
        raise HTTPException(400, "Only WAV accepted")

    content = await file.read()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = subprocess.run(
            [str(WHISPER_CLI), "-m", str(WHISPER_MODEL), "-f", tmp_path, "--no-timestamps"],
            capture_output=True, text=True
        )

        if result.returncode != 0:
            raise HTTPException(500, result.stderr)

        text = result.stdout.strip()
        return {"text": text}

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)