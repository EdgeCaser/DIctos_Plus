import os
import whisper
import torch

print("Starting Whisper test...")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

audio_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "audio", "250506_1504.mp3")
print(f"\nLoading audio file: {audio_path}")
print(f"File exists: {os.path.exists(audio_path)}")
print(f"File size: {os.path.getsize(audio_path)} bytes")

print("\nLoading Whisper model...")
model = whisper.load_model("base", device="cuda" if torch.cuda.is_available() else "cpu")
print("Model loaded successfully")

print("\nStarting transcription...")
try:
    # Load just the first 30 seconds
    result = model.transcribe(
        audio_path,
        language="en",
        initial_prompt="",
        verbose=True
    )
    print("\nTranscription successful!")
    print("\nTranscribed text:")
    print(result["text"])
except Exception as e:
    print(f"\nError during transcription: {str(e)}")
    print(f"Error type: {type(e)}")
