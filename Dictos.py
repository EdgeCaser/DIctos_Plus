import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import time
import torch
import os
import sys
import shutil

# Set ffmpeg path
ffmpeg_path = r"C:\Users\ianfe\OneDrive\Documents\GitHub\audioWhisper\ffmpeg-2025-05-05-git-f4e72eb5a3-full_build\ffmpeg-2025-05-05-git-f4e72eb5a3-full_build\bin"
if not os.path.exists(ffmpeg_path):
    messagebox.showerror("Error", f"ffmpeg directory not found at: {ffmpeg_path}")
    sys.exit(1)

# Add ffmpeg to PATH
os.environ["PATH"] = ffmpeg_path + os.pathsep + os.environ["PATH"]

# Verify ffmpeg is accessible
try:
    import subprocess
    result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
    if result.returncode != 0:
        messagebox.showerror("Error", "ffmpeg is not working properly. Please check your installation.")
        sys.exit(1)
except Exception as e:
    messagebox.showerror("Error", f"Failed to verify ffmpeg: {str(e)}")
    sys.exit(1)

import json
import queue
import threading
import numpy as np
import soundfile as sf
from datetime import datetime
from transformers import pipeline
from pyannote.audio import Pipeline as PyannotePipeline
from dotenv import load_dotenv

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load environment variables
load_dotenv()

# Ask user for Hugging Face token (or set as env var)
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    import getpass
    HF_TOKEN = getpass.getpass("Enter your Hugging Face token (for pyannote.audio): ")

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

class TranscriptionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Audio Transcription with Speaker Diarization (HuggingFace + pyannote)")
        self.root.geometry("600x600")
        self.audio_path = tk.StringVar()
        self.audio_duration = tk.DoubleVar(value=0.0)
        self.max_duration_var = tk.StringVar()
        self.num_speakers_var = tk.StringVar(value="2")
        self.status_queue = queue.Queue()

        tk.Label(self.root, text="Audio Transcription Tool", font=("Arial", 16)).pack(pady=10)
        tk.Label(self.root, text="Audio File:").pack()
        tk.Entry(self.root, textvariable=self.audio_path, width=50, state='readonly').pack()
        tk.Button(self.root, text="Select Audio File", command=self.select_file).pack(pady=5)
        tk.Label(self.root, text="Duration to Process (seconds, 0 for full audio):").pack()
        tk.Entry(self.root, textvariable=self.max_duration_var, width=20).pack()
        tk.Label(self.root, text="Estimated Number of Speakers (leave blank for auto):").pack()
        tk.Entry(self.root, textvariable=self.num_speakers_var, width=20).pack()
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
        file_path = filedialog.askopenfilename(
            title="Select Audio File to Transcribe",
            filetypes=[("Audio Files", "*.mp3 *.wav *.m4a")]
        )
        if file_path:
            self.audio_path.set(file_path)
            duration = self.get_audio_duration(file_path)
            if duration == 0.0:
                messagebox.showerror("Error", "Failed to load audio file.")
                return
            self.audio_duration.set(duration)
            self.max_duration_var.set(str(duration))
            self.update_status(f"Selected audio file: {file_path}\nDuration: {duration:.1f} seconds ({duration/60:.1f} minutes)")

    def get_audio_duration(self, file_path: str) -> float:
        try:
            info = sf.info(file_path)
            return info.duration
        except Exception as e:
            self.update_status(f"Error determining audio duration: {str(e)}")
            return 0.0

    def update_status(self, message: str):
        self.status_queue.put(message)

    def check_status_queue(self):
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
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter valid numbers for duration and number of speakers.")
            return

        self.start_button.config(state='disabled')
        self.progress['value'] = 0
        self.update_status("Starting transcription process...")

        thread = threading.Thread(target=self.run_transcription, args=(
            self.audio_path.get(), max_duration, num_speakers
        ))
        thread.daemon = True
        thread.start()

    def run_transcription(self, file_path: str, max_duration: float, num_speakers: int):
        try:
            self.progress['value'] = 5
            self.root.update()

            # Preprocessing audio
            self.update_status("Preprocessing audio...")
            audio, sr = sf.read(file_path)
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            if max_duration > 0:
                samples = int(max_duration * sr)
                audio = audio[:samples]
            duration = len(audio) / sr

            # Prepare temp_chunks directory
            temp_chunks_dir = os.path.join(os.path.dirname(file_path), "temp_chunks")
            os.makedirs(temp_chunks_dir, exist_ok=True)

            chunk_duration = 300  # seconds (5 minutes)
            overlap = 1  # seconds
            num_chunks = int(np.ceil(duration / chunk_duration))
            self.update_status(f"Audio will be split into {num_chunks} chunks of {chunk_duration} seconds each (with {overlap}s overlap).")

            asr_pipe = pipeline(
                "automatic-speech-recognition",
                model="openai/whisper-small",
                device=0 if device == "cuda" else -1,
                return_timestamps="word"
            )

            words = []
            for i in range(num_chunks):
                start_sample = int(max(0, (i * chunk_duration - overlap) * sr))
                end_sample = int(min((i + 1) * chunk_duration * sr, len(audio)))
                chunk_audio = audio[start_sample:end_sample]
                temp_chunk_wav = os.path.join(temp_chunks_dir, f"temp_chunk_{i}.wav")
                sf.write(temp_chunk_wav, chunk_audio, sr)
                self.update_status(f"Transcribing chunk {i+1}/{num_chunks}...")
                chunk_result = asr_pipe(temp_chunk_wav)
                chunk_words = chunk_result["chunks"]
                chunk_offset = max(0, i * chunk_duration - overlap)
                for word in chunk_words:
                    if word["timestamp"] is not None and all(t is not None for t in word["timestamp"]):
                        word["timestamp"] = [t + chunk_offset for t in word["timestamp"]]
                        words.append(word)
                self.progress['value'] = 20 + int(30 * (i + 1) / num_chunks)
                self.root.update()
                # temp files will be deleted after all chunks are processed

            # Clean up temp_chunks directory
            shutil.rmtree(temp_chunks_dir, ignore_errors=True)

            self.progress['value'] = 50
            self.root.update()

            # Diarization (still on full audio)
            self.update_status("Running speaker diarization with pyannote.audio...")
            temp_wav = os.path.join(os.path.dirname(file_path), "temp_for_transcribe.wav")
            sf.write(temp_wav, audio, sr)
            diarization_pipeline = PyannotePipeline.from_pretrained(
                "pyannote/speaker-diarization",
                use_auth_token=HF_TOKEN
            )
            diarization = diarization_pipeline(temp_wav, num_speakers=num_speakers)
            self.progress['value'] = 80
            self.root.update()

            # Assign speakers to words
            speaker_segments = {}
            for turn in diarization.itertracks(yield_label=True):
                segment, _, speaker = turn
                if speaker not in speaker_segments:
                    speaker_segments[speaker] = []
                speaker_segments[speaker].append({"start": segment.start, "end": segment.end})

            for word in words:
                word_start = word["timestamp"][0]
                word_end = word["timestamp"][1]
                word["speaker"] = "UNKNOWN"
                for speaker, segments in speaker_segments.items():
                    for seg in segments:
                        if word_start >= seg["start"] and word_end <= seg["end"]:
                            word["speaker"] = speaker
                            break
                    if word["speaker"] != "UNKNOWN":
                        break

            self.progress['value'] = 90
            self.root.update()

            # Deduplicate repeated words at chunk boundaries
            deduped_words = []
            for i, word in enumerate(words):
                if i > 0:
                    prev_word = deduped_words[-1]
                    # Check if text matches and timestamps are close (within 2 seconds)
                    if (
                        word["text"].strip().lower() == prev_word["text"].strip().lower() and
                        abs(word["timestamp"][0] - prev_word["timestamp"][1]) < 2
                    ):
                        continue  # skip duplicate
                deduped_words.append(word)
            words = deduped_words

            # Save output
            self.update_status("Saving transcription results...")
            txt_output, json_output = self.save_transcription(file_path, words, speaker_segments)
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

    def save_transcription(self, file_path: str, words, speaker_segments):
        os.makedirs('output', exist_ok=True)
        filename = os.path.basename(file_path)
        name_without_ext = os.path.splitext(filename)[0]
        txt_output = os.path.join('output', f"{name_without_ext}_transcript.txt")
        json_output = os.path.join('output', f"{name_without_ext}_transcript.json")

        with open(txt_output, 'w', encoding='utf-8') as f:
            self.update_status("Transcription result with speakers:")
            self.update_status("-" * 50)
            for word in words:
                timestamp = self.format_time(word["timestamp"][0])
                speaker = word.get("speaker", "UNKNOWN")
                text = word["text"].strip()
                output_line = f"[{timestamp}] {speaker}: {text}"
                self.update_status(output_line)
                f.write(output_line + '\n')
            self.update_status("-" * 50)

        output_data = {"words": words, "speaker_segments": speaker_segments}
        with open(json_output, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2)

        return txt_output, json_output

    def format_time(self, seconds: float) -> str:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"

if __name__ == "__main__":
    root = tk.Tk()
    app = TranscriptionApp(root)
    root.mainloop()