import os
import wave

audio_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "audio", "250506_1504.mp3")
print(f"Checking file: {audio_path}")
print(f"File exists: {os.path.exists(audio_path)}")
print(f"File size: {os.path.getsize(audio_path)} bytes")

# Try to read the first few bytes of the file
try:
    with open(audio_path, 'rb') as f:
        header = f.read(10)
        print(f"First 10 bytes: {header.hex()}")
except Exception as e:
    print(f"Error reading file: {str(e)}")
