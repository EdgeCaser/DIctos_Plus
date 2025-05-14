import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import time
from typing import Dict, List
import torch
import os
import json
from datetime import datetime
import tempfile
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, lfilter
import soundfile as sf
import threading
import queue
import subprocess
import sys
import warnings

# Suppress warnings to clean up output
warnings.filterwarnings("ignore")

# Patch numpy.NaN to numpy.nan for compatibility
np.NaN = np.nan

# Import whisperx after numpy patch
import whisperx

# Configure ffmpeg paths
ffmpeg_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                         'ffmpeg-2025-05-05-git-f4e72eb5a3-full_build',
                         'ffmpeg-2025-05-05-git-f4e72eb5a3-full_build',
                         'bin'))
ffmpeg_path = os.path.join(ffmpeg_dir, 'ffmpeg.exe')
ffprobe_path = os.path.join(ffmpeg_dir, 'ffprobe.exe')

# Debug: Print paths and existence
print(f"Looking for ffmpeg in: {ffmpeg_dir}")
print(f"Directory exists: {os.path.exists(ffmpeg_dir)}")
print(f"ffmpeg exists: {os.path.exists(ffmpeg_path)}")
print(f"ffprobe exists: {os.path.exists(ffprobe_path)}")

# Add ffmpeg directory to system PATH
if ffmpeg_dir not in os.environ['PATH']:
    os.environ['PATH'] = ffmpeg_dir + os.pathsep + os.environ['PATH']

# Verify that the executables exist
if not os.path.exists(ffmpeg_path) or not os.path.exists(ffprobe_path):
    raise RuntimeError(f"ffmpeg executables not found in {ffmpeg_dir}")

# Set environment variables for ffmpeg
os.environ["FFMPEG_BINARY"] = ffmpeg_path
os.environ["FFPROBE_BINARY"] = ffprobe_path

class TranscriptionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Audio Transcription with Speaker Diarization")
        self.root.geometry("600x600")  # Increased height to accommodate new fields
        
        # Variables
        self.audio_path = tk.StringVar()
        self.audio_duration = tk.DoubleVar(value=0.0)
        self.max_duration_var = tk.StringVar()
        self.num_speakers_var = tk.StringVar(value="2")
        self.max_new_tokens_var = tk.StringVar(value="128")  # Default value
        self.clip_timestamps_var = tk.StringVar(value="")  # Default empty (None)
        self.hallucination_silence_threshold_var = tk.StringVar(value="")  # Default empty (None)
        self.hotwords_var = tk.StringVar(value="")  # Default empty (None)
        self.status_queue = queue.Queue()
        
        # Debug: Log the ffmpeg paths
        print(f"FFMPEG_BINARY: {os.environ.get('FFMPEG_BINARY')}")
        print(f"FFPROBE_BINARY: {os.environ.get('FFPROBE_BINARY')}")
        
        # GUI Elements
        tk.Label(self.root, text="Audio Transcription Tool", font=("Arial", 16)).pack(pady=10)
        
        tk.Label(self.root, text="Audio File:").pack()
        tk.Entry(self.root, textvariable=self.audio_path, width=50, state='readonly').pack()
        tk.Button(self.root, text="Select Audio File", command=self.select_file).pack(pady=5)
        
        tk.Label(self.root, text="Duration to Process (seconds, 0 for full audio):").pack()
        tk.Entry(self.root, textvariable=self.max_duration_var, width=20).pack()
        
        tk.Label(self.root, text="Estimated Number of Speakers (leave blank for auto):").pack()
        tk.Entry(self.root, textvariable=self.num_speakers_var, width=20).pack()
        
        # User options (not TranscriptionOptions, just for future use)
        tk.Label(self.root, text="Max New Tokens (default 128, adjust for performance):").pack()
        tk.Entry(self.root, textvariable=self.max_new_tokens_var, width=20).pack()
        
        tk.Label(self.root, text="Clip Timestamps (e.g., '0-10,20-30', leave blank for full):").pack()
        tk.Entry(self.root, textvariable=self.clip_timestamps_var, width=20).pack()
        
        tk.Label(self.root, text="Hallucination Silence Threshold (seconds, leave blank for default):").pack()
        tk.Entry(self.root, textvariable=self.hallucination_silence_threshold_var, width=20).pack()
        
        tk.Label(self.root, text="Hotwords (comma-separated, e.g., 'meeting,project', leave blank for none):").pack()
        tk.Entry(self.root, textvariable=self.hotwords_var, width=20).pack()
        
        self.start_button = tk.Button(self.root, text="Start Transcription", command=self.start_transcription)
        self.start_button.pack(pady=10)
        
        tk.Label(self.root, text="Progress:").pack()
        self.progress = ttk.Progressbar(self.root, length=400, mode='determinate')
        self.progress.pack()
        
        tk.Label(self.root, text="Status:").pack()
        self.status_text = tk.Text(self.root, height=10, width=60, state='disabled')
        self.status_text.pack(pady=5)
        
        self.root.after(100, self.check_status_queue)

    def select_file(self):
        """Show file picker dialog and set audio path."""
        file_path = filedialog.askopenfilename(
            title="Select Audio File to Transcribe",
            filetypes=[("Audio Files", "*.mp3 *.wav *.m4a")]
        )
        if file_path:
            self.audio_path.set(file_path)
            duration = self.get_audio_duration(file_path)
            if duration == 0.0:
                messagebox.showerror("Error", "Failed to load audio file. Ensure ffmpeg is installed and in your PATH.")
                return
            self.audio_duration.set(duration)
            self.max_duration_var.set(str(duration))
            self.update_status(f"Selected audio file: {file_path}\nDuration: {duration:.1f} seconds ({duration/60:.1f} minutes)")

    def get_audio_duration(self, file_path: str) -> float:
        """Get the duration of the audio file in seconds."""
        try:
            info = sf.info(file_path)
            return info.duration
        except Exception as e:
            self.update_status(f"Error determining audio duration: {str(e)}")
            return 0.0

    def update_status(self, message: str):
        """Update the status text area with a new message."""
        self.status_queue.put(message)

    def check_status_queue(self):
        """Check the status queue and update the GUI."""
        try:
            while True:
                message = self.status_queue.get_nowait()
                self.status_text.config(state='normal')
                self.status_text.insert(tk.END, message + "\n")
                self.status_text.see(tk.END)
                self.status_text.config(state='disabled')
        except queue.Empty:
            pass
        self.root.after(100, self.check_status_queue)

    def start_transcription(self):
        """Start the transcription process in a separate thread."""
        try:
            max_duration = float(self.max_duration_var.get())
            if max_duration < 0:
                messagebox.showerror("Invalid Input", "Please enter a non-negative number for duration (or 0 for full audio).")
                return
            if max_duration > self.audio_duration.get():
                messagebox.showerror("Invalid Input", f"Duration cannot exceed audio length ({self.audio_duration.get():.1f} seconds).")
                return
            num_speakers_input = self.num_speakers_var.get().strip()
            num_speakers = int(num_speakers_input) if num_speakers_input else None
            if num_speakers is not None and num_speakers <= 0:
                messagebox.showerror("Invalid Input", "Number of speakers must be a positive integer (or leave blank for auto-detection).")
                return
            
            # Validate new transcription options
            max_new_tokens_input = self.max_new_tokens_var.get().strip()
            try:
                max_new_tokens = int(max_new_tokens_input) if max_new_tokens_input else 128
                if max_new_tokens <= 0:
                    raise ValueError
            except ValueError:
                messagebox.showerror("Invalid Input", "Max New Tokens must be a positive integer (or leave blank for default 128).")
                return

            clip_timestamps_input = self.clip_timestamps_var.get().strip()
            clip_timestamps = None
            if clip_timestamps_input:
                try:
                    # Parse clip_timestamps as a list of start-end pairs, e.g., "0-10,20-30"
                    clip_timestamps = []
                    for pair in clip_timestamps_input.split(","):
                        start, end = map(float, pair.split("-"))
                        if start < 0 or end <= start:
                            raise ValueError
                        clip_timestamps.append((start, end))
                except ValueError:
                    messagebox.showerror("Invalid Input", "Clip Timestamps must be in format 'start-end,start-end' (e.g., '0-10,20-30').")
                    return

            hallucination_silence_threshold_input = self.hallucination_silence_threshold_var.get().strip()
            hallucination_silence_threshold = None
            if hallucination_silence_threshold_input:
                try:
                    hallucination_silence_threshold = float(hallucination_silence_threshold_input)
                    if hallucination_silence_threshold <= 0:
                        raise ValueError
                except ValueError:
                    messagebox.showerror("Invalid Input", "Hallucination Silence Threshold must be a positive number (or leave blank for default).")
                    return

            hotwords_input = self.hotwords_var.get().strip()
            hotwords = None
            if hotwords_input:
                hotwords = hotwords_input.split(",")
                hotwords = [word.strip() for word in hotwords if word.strip()]

        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter valid numbers for duration and number of speakers.")
            return

        if max_duration > 3600:
            confirm = messagebox.askyesno(
                "Warning",
                f"You've selected to process {max_duration:.1f} seconds ({max_duration/60:.1f} minutes).\n"
                "This may take significant time and resources.\nContinue?"
            )
            if not confirm:
                self.update_status("User canceled long duration processing.")
                return

        self.start_button.config(state='disabled')
        self.progress['value'] = 0
        self.update_status("Starting transcription process...")

        thread = threading.Thread(target=self.run_transcription, args=(
            self.audio_path.get(), max_duration, num_speakers, 
            max_new_tokens, clip_timestamps, hallucination_silence_threshold, hotwords
        ))
        thread.daemon = True
        thread.start()

    def run_transcription(self, file_path: str, max_duration: float, num_speakers: int, 
                         max_new_tokens: int, clip_timestamps: list, 
                         hallucination_silence_threshold: float, hotwords: list):
        """Run the transcription and diarization process with WhisperX."""
        try:
            self.progress['value'] = 5
            self.root.update()

            # Preprocess audio
            self.update_status("Preprocessing audio...")
            duration_ms = int(max_duration * 1000) if max_duration > 0 else None
            preprocessed_path = self.preprocess_audio(file_path, duration_ms)
            if not preprocessed_path:
                messagebox.showerror("Error", "Failed to preprocess audio. Ensure ffmpeg is installed and in your PATH.")
                return
            self.progress['value'] = 25
            self.root.update()

            # Load WhisperX model
            self.update_status("Loading WhisperX model (base.en)...")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            start_time = time.time()
            
            try:
                # Create ASR options dictionary
                asr_options = {
                    "multilingual": False,
                    "max_new_tokens": max_new_tokens,
                    "clip_timestamps": clip_timestamps,
                    "hallucination_silence_threshold": hallucination_silence_threshold,
                    "hotwords": hotwords
                }
                
                self.update_status("Attempting to download/load model...")
                # Print cache directory for debugging
                cache_dir = os.path.expanduser('~/.cache/whisperx')
                self.update_status(f"Model cache directory: {cache_dir}")
                self.update_status(f"Cache directory exists: {os.path.exists(cache_dir)}")
                
                # Load model with correct parameters and more detailed error handling
                try:
                    model = whisperx.load_model(
                        "base.en",
                        device,
                        compute_type="float32",
                        asr_options=asr_options,
                        language="en",
                        download_root=cache_dir  # Explicitly set download root
                    )
                except Exception as model_error:
                    self.update_status(f"Detailed model loading error: {str(model_error)}")
                    self.update_status("Attempting to load model with fallback options...")
                    # Try with minimal options as fallback, but still provide asr_options
                    model = whisperx.load_model(
                        "base.en",
                        device,
                        compute_type="float32",
                        asr_options=asr_options,
                        language="en"
                    )
                
                self.update_status(f"Model loaded in {time.time() - start_time:.2f} seconds")
            except Exception as e:
                self.update_status(f"Error during model loading: {str(e)}")
                self.update_status("Full error details:")
                import traceback
                self.update_status(traceback.format_exc())
                raise

            # Transcribe with diarization
            self.update_status("Transcribing and diarizing with WhisperX...")
            audio = whisperx.load_audio(preprocessed_path)
            result = model.transcribe(
                audio,
                batch_size=32,
                language='en',  # Since we're using medium.en model
                print_progress=True
            )
            self.progress['value'] = 50
            self.root.update()

            # Align segments
            self.update_status("Aligning transcription segments...")
            align_model, metadata = whisperx.load_align_model(language_code="en", device=device)
            result = whisperx.align(result["segments"], align_model, metadata, audio, device)
            self.progress['value'] = 70
            self.root.update()

            # Diarize
            self.update_status("Performing speaker diarization...")
            diarizer = whisperx.DiarizationPipeline(use_auth_token=None, device=device)
            diarize_segments = diarizer(audio, min_speakers=1, max_speakers=num_speakers if num_speakers else 10)
            result = whisperx.assign_word_speakers(diarize_segments, result)
            
            # Convert to script's speaker segments format
            speaker_segments = {}
            for segment in result["segments"]:
                speaker = segment.get("speaker", "UNKNOWN")
                if speaker not in speaker_segments:
                    speaker_segments[speaker] = []
                speaker_segments[speaker].append({
                    "start": segment["start"],
                    "end": segment["end"]
                })
            
            self.progress['value'] = 90
            self.root.update()

            # Save output
            self.update_status("Saving transcription results...")
            txt_output, json_output = self.save_transcription(file_path, result, speaker_segments)
            self.progress['value'] = 100
            self.root.update()

            self.update_status("Transcription complete!")
            messagebox.showinfo(
                "Transcription Complete",
                f"Transcription completed and saved to:\n{txt_output}\n\nDetailed JSON output saved to:\n{json_output}"
            )

        except Exception as e:
            self.update_status(f"Error: {str(e)}")
            messagebox.showerror(
                "Error",
                f"An error occurred: {str(e)}"
            )
        finally:
            self.start_button.config(state='normal')

    def preprocess_audio(self, input_path: str, duration_ms: int = None) -> str:
        """Preprocess audio file: trim (if specified), normalize, convert to mono WAV, and apply noise reduction."""
        try:
            # Read audio file using soundfile
            data, sample_rate = sf.read(input_path)
            
            # Convert to mono if stereo
            if len(data.shape) > 1:
                data = np.mean(data, axis=1)
            
            # Trim if duration specified
            if duration_ms and duration_ms > 0:
                samples = int((duration_ms / 1000) * sample_rate)
                if samples < len(data):
                    data = data[:samples]
                self.update_status(f"Preprocessing audio for {duration_ms/1000:.1f} seconds...")
            else:
                self.update_status(f"Preprocessing full audio ({len(data)/sample_rate:.1f} seconds)...")
            
            # Apply high-pass filter
            def high_pass_filter(data, cutoff=150, fs=44100, order=5):
                nyq = 0.5 * fs
                normal_cutoff = cutoff / nyq
                b, a = butter(order, normal_cutoff, btype='high', analog=False)
                return lfilter(b, a, data)
            
            filtered_data = high_pass_filter(data, fs=sample_rate)
            
            # Normalize audio
            if filtered_data.max() != 0:
                filtered_data = filtered_data / np.abs(filtered_data).max()
            
            # Save to temp file
            temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp')
            os.makedirs(temp_dir, exist_ok=True)
            output_path = os.path.join(temp_dir, 'preprocessed_audio.wav')
            
            # Save with 16kHz sample rate
            sf.write(output_path, filtered_data, 16000)
            
            self.update_status(f"Preprocessed audio saved to: {output_path}")
            return output_path
        except Exception as e:
            self.update_status(f"Error preprocessing audio: {str(e)}")
            return None

    def ensure_output_dir(self):
        """Create output directory if it doesn't exist."""
        os.makedirs('output', exist_ok=True)

    def get_output_path(self, input_path: str, suffix: str) -> str:
        """Generate output file path in the output directory."""
        filename = os.path.basename(input_path)
        name_without_ext = os.path.splitext(filename)[0]
        return os.path.join('output', f"{name_without_ext}{suffix}")

    def save_transcription(self, file_path: str, result, speaker_segments):
        """Save the transcription and speaker segments to files."""
        self.ensure_output_dir()
        txt_output = self.get_output_path(file_path, '_transcript.txt')
        json_output = self.get_output_path(file_path, '_transcript.json')
        
        with open(txt_output, 'w', encoding='utf-8') as f:
            self.update_status("Transcription result with speakers:")
            self.update_status("-" * 50)
            for segment in result["segments"]:
                segment_start = segment["start"]
                segment_end = segment["end"]
                current_speaker = None
                
                for speaker, segments in speaker_segments.items():
                    for s in segments:
                        if not (segment_end < s["start"] or segment_start > s["end"]):
                            current_speaker = speaker
                            break
                    if current_speaker:
                        break
                
                if not current_speaker:
                    min_distance = float('inf')
                    for speaker, segments in speaker_segments.items():
                        for s in segments:
                            distance = min(abs(segment_start - s["end"]), abs(segment_end - s["start"]))
                            if distance < min_distance:
                                min_distance = distance
                                current_speaker = speaker
                    if not current_speaker:
                        current_speaker = "UNKNOWN"
                        self.update_status(f"No speaker for segment {segment_start:.2f}s - {segment_end:.2f}s")
                
                timestamp = self.format_time(segment_start)
                text = segment["text"].strip()
                output_line = f"[{timestamp}] {current_speaker}: {text}"
                self.update_status(output_line)
                f.write(output_line + '\n')
            self.update_status("-" * 50)
        
        output_data = {"transcription": result, "speaker_segments": speaker_segments}
        with open(json_output, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2)
        
        return txt_output, json_output

    def format_time(self, seconds: float) -> str:
        """Convert seconds to a MM:SS timestamp."""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"

if __name__ == "__main__":
    print("Starting audio transcription program with GUI...")
    
    # Print diagnostic information
    print("\nDiagnostic Information:")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"cuDNN version: {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'Not available'}")
        print(f"cuDNN enabled: {torch.backends.cudnn.enabled}")
    
    import pkg_resources
    print("\nRelevant package versions:")
    for package in ['torch', 'pyannote.audio', 'whisperx']:
        try:
            version = pkg_resources.get_distribution(package).version
            print(f"{package}: {version}")
        except pkg_resources.DistributionNotFound:
            print(f"{package}: Not found")
    
    # Configure CUDA and torch settings
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.empty_cache()
        
        # Enable TF32 for better performance
        if torch.cuda.get_device_capability()[0] >= 8:  # Ampere or newer GPUs
            print("Enabling TF32 for better performance...")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        # Set device to GPU
        device = torch.device("cuda")
    else:
        print("CUDA is not available. Using CPU.")
        device = torch.device("cpu")
    
    try:
        # Initialize GUI
        root = tk.Tk()
        app = TranscriptionApp(root)
        root.mainloop()
    except Exception as e:
        print(f"Error: {str(e)}")
        if "cudnn" in str(e).lower():
            print("\nCUDA/cuDNN Error: Please ensure you have:")
            print("1. Installed CUDA Toolkit matching your PyTorch version")
            print("2. Installed cuDNN matching your CUDA version")
            print("3. Added CUDA and cuDNN directories to your system PATH")
            print("\nYou may need to:")
            print("1. Download cuDNN from NVIDIA website")
            print("2. Extract and copy cudnn64_8.dll to your CUDA installation's bin directory")
            print("3. Restart your computer")
        sys.exit(1)