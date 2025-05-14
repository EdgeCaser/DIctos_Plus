# Audio Transcription with Whisper and Speaker Diarization

This project uses OpenAI's Whisper model for transcription and Pyannote for speaker diarization to create detailed transcripts with speaker identification.

## Prerequisites

1. Python 3.11 (Python 3.13 is not supported yet)
2. CUDA-capable GPU (for optimal performance)
3. ffmpeg installed and in PATH

## Setup Instructions

1. Clone this repository
2. Run the setup batch file:
```bash
run_dictos.bat
```
This will:
- Create a Python virtual environment
- Install all required dependencies
- Launch the application

## Manual Setup (if not using batch file)

1. Create and activate a Python 3.11 virtual environment:
```bash
py -3.11 -m venv venv
venv\Scripts\activate
```

2. Install dependencies:
```bash
python -m pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## Usage

Run the main script:
```bash
python DictosPlus.py
```

## Features

- Audio transcription using Whisper
- Speaker diarization using Pyannote
- Support for multiple audio formats
- Automatic punctuation and formatting
```
