import os
import whisper
import torch
import subprocess
import tempfile

print("Starting Whisper test...")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Original audio path
orig_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "audio", "250506_1504.mp3")
print(f"\nOriginal audio file: {orig_path}")
print(f"File exists: {os.path.exists(orig_path)}")
print(f"File size: {os.path.getsize(orig_path)} bytes")

# Create temp directory
temp_dir = tempfile.mkdtemp()
temp_wav = os.path.join(temp_dir, "audio.wav")

# Convert to WAV using ffmpeg
print("\nConverting to WAV...")
try:
    subprocess.run([
        "ffmpeg", "-i", orig_path,
        "-ar", "16000",  # 16kHz sample rate
        "-ac", "1",      # mono
        "-c:a", "pcm_s16le",  # 16-bit PCM
        temp_wav
    ], check=True, capture_output=True)
    print("Conversion successful")
    print(f"WAV file exists: {os.path.exists(temp_wav)}")
    print(f"WAV file size: {os.path.getsize(temp_wav)} bytes")
except subprocess.CalledProcessError as e:
    print(f"FFmpeg error: {e.stderr.decode()}")
    raise

print("\nLoading Whisper model...")
model = whisper.load_model("base", device="cuda" if torch.cuda.is_available() else "cpu")
print("Model loaded successfully")

print("\nStarting transcription...")
try:
    result = model.transcribe(
        temp_wav,
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
finally:
    # Clean up
    try:
        import shutil
        shutil.rmtree(temp_dir)
        print("\nCleaned up temporary files")
    except:
        pass
