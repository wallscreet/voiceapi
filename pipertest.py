# from piper.voice import PiperVoice
# import io

# PIPER_MODEL = "models/piper/en_US-hfc_female-medium.onnx"  # your path
# voice = PiperVoice.load(PIPER_MODEL)

# test_texts = [
#     "Why did the astronaut break up with his girlfriend? Because he needed space!",
#     "Why don't skeletons fight each other? They don't have the guts!",
#     "Hello this is a simple test sentence without jokes or punctuation issues.",
#     "Testing Piper TTS one two three four five."
# ]

# for i, t in enumerate(test_texts):
#     wav_io = io.BytesIO()
#     voice.synthesize(t, wav_io)
#     wav_io.seek(0)
#     bytes_out = len(wav_io.read())
#     print(f"Text {i+1}: '{t}' → {bytes_out} bytes")

import wave
from piper import PiperVoice

voice = PiperVoice.load("models/piper/en_US-hfc_female-medium.onnx")
with wave.open("test.wav", "wb") as wav_file:
    voice.synthesize_wav("Welcome to the world of speech synthesis!", wav_file)