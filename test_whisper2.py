import os
import whisper
import torch
import shutil
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

# Create temp directory and copy file
temp_dir = tempfile.mkdtemp()
temp_path = os.path.join(temp_dir, "audio.mp3")
shutil.copy2(orig_path, temp_path)
print(f"\nCopied to temp file: {temp_path}")
print(f"Temp file exists: {os.path.exists(temp_path)}")
print(f"Temp file size: {os.path.getsize(temp_path)} bytes")

print("\nLoading Whisper model...")
model = whisper.load_model("base", device="cuda" if torch.cuda.is_available() else "cpu")
print("Model loaded successfully")

print("\nStarting transcription...")
try:
    # Load just the first 30 seconds
    result = model.transcribe(
        temp_path,
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
        shutil.rmtree(temp_dir)
        print("\nCleaned up temporary files")
    except:
        pass
