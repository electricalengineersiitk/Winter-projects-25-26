import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
from openai import OpenAI
from dotenv import load_dotenv
import os
import tempfile

load_dotenv()

groq_client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

SAMPLE_RATE = 16000  # 16kHz, optimal for Whisper


def record_audio(duration=5):
    """Record audio from microphone for the given duration (seconds)."""
    print(f"\n Recording for {duration} seconds... Speak now!")
    audio = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
    sd.wait()
    print("Recording complete!")
    return audio


def transcribe_audio(audio):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav.write(tmp.name, SAMPLE_RATE, audio)
        tmp_path = tmp.name

    try:
        print("Transcribing with Whisper...")
        with open(tmp_path, "rb") as audio_file:
            transcription = groq_client.audio.transcriptions.create(
                model="whisper-large-v3",
                file=audio_file,
                language="en"
            )
        text = transcription.text.strip()
        print(f"You said: \"{text}\"")
        return text
    finally:
        os.unlink(tmp_path)


def get_voice_command(duration=5):
    audio = record_audio(duration)
    text = transcribe_audio(audio)
    return text


if __name__ == "__main__":
    print("=" * 50)
    print("VOICE INPUT TEST - Groq Whisper")
    print("=" * 50)
    print("\nThis will record your voice and transcribe it.")
    print("Make sure your microphone is connected.\n")

    duration = int(input("Recording duration in seconds (default 5): ").strip() or "5")
    command = get_voice_command(duration)

    print(f"\nFinal transcription: \"{command}\"")
    print("\nThis text would be sent to the LLM for waypoint generation.")
    print("Voice input test complete!")
