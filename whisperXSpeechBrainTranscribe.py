import whisperx
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import time
from typing import Dict, List
import torch
import os
import json
from datetime import datetime
import tempfile
from pydub import AudioSegment
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, lfilter
import soundfile as sf
import threading
import queue
from speechbrain.inference.diarization import SpeechDiarizer

class TranscriptionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Audio Transcription with Speaker Diarization")
        self.root.geometry("600x500")
        
        # Variables
        self.audio_path = tk.StringVar()
        self.audio_duration = tk.DoubleVar(value=0.0)
        self.max_duration_var = tk.StringVar()
        self.num_speakers_var = tk.StringVar(value="2")
        self.status_queue = queue.Queue()
        
        # GUI Elements
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
        """Show file picker dialog and set audio path."""
        file_path = filedialog.askopenfilename(
            title="Select Audio File to Transcribe",
            filetypes=[("Audio Files", "*.mp3 *.wav *.m4a")]
        )
        if file_path:
            self.audio_path.set(file_path)
            duration = self.get_audio_duration(file_path)
            self.audio_duration.set(duration)
            self.max_duration_var.set(str(duration))
            self.update_status(f"Selected audio file: {file_path}\nDuration: {duration:.1f} seconds ({duration/60:.1f} minutes)")

    def get_audio_duration(self, file_path: str) -> float:
        """Get the duration of the audio file in seconds."""
        try:
            audio = AudioSegment.from_file(file_path)
            duration_seconds = len(audio) / 1000
            return duration_seconds
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

        thread = threading.Thread(target=self.run_transcription, args=(self.audio_path.get(), max_duration, num_speakers))
        thread.daemon = True
        thread.start()

    def run_transcription(self, file_path: str, max_duration: float, num_speakers: int):
        """Run the transcription with WhisperX and diarization with SpeechBrain."""
        try:
            self.progress['value'] = 5
            self.root.update()

            # Preprocess audio
            self.update_status("Preprocessing audio...")
            duration_ms = int(max_duration * 1000) if max_duration > 0 else None
            preprocessed_path = self.preprocess_audio(file_path, duration_ms)
            self.progress['value'] = 25
            self.root.update()

            # Load WhisperX model for transcription
            self.update_status("Loading WhisperX model (medium.en)...")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            start_time = time.time()
            model = whisperx.load_model("medium.en", device, compute_type="float32")
            self.update_status(f"Model loaded in {time.time() - start_time:.2f} seconds")
            self.progress['value'] = 30
            self.root.update()

            # Transcribe
            self.update_status("Transcribing with WhisperX...")
            audio = whisperx.load_audio(preprocessed_path)
            result = model.transcribe(audio, batch_size=32)
            self.progress['value'] = 50
            self.root.update()

            # Align segments
            self.update_status("Aligning transcription segments...")
            align_model, metadata = whisperx.load_align_model(language_code="en", device=device)
            result = whisperx.align(result["segments"], align_model, metadata, audio, device)
            self.progress['value'] = 70
            self.root.update()

            # Diarize with SpeechBrain
            self.update_status("Performing speaker diarization with SpeechBrain...")
            speaker_segments = self.get_speaker_segments_speechbrain(preprocessed_path, max_duration, num_speakers)
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

    def get_speaker_segments_speechbrain(self, file_path: str, max_duration: float = None, num_speakers: int = None) -> Dict[str, List[dict]]:
        """Perform speaker diarization using SpeechBrain 1.0.0 API."""
        self.update_status("Detecting speakers with SpeechBrain...")
        audio = AudioSegment.from_file(file_path)
        audio_duration = len(audio) / 1000
        if max_duration and max_duration > 0:
            self.update_status(f"Processing first {max_duration:.1f} seconds (out of {audio_duration:.1f} seconds total)...")
        else:
            self.update_status(f"Processing full audio ({audio_duration:.1f} seconds)...")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Use SpeechBrain's new diarization pipeline
        diarizer = SpeechDiarizer.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="speechbrain_models",
            run_opts={"device": device}
        )
        
        # Perform diarization
        diarization_result = diarizer.diarize_file(file_path, duration=max_duration)
        
        # Convert diarization result to script's format
        speaker_segments = {}
        for turn, speaker in diarization_result.items():
            start, end = turn
            if speaker not in speaker_segments:
                speaker_segments[speaker] = []
            speaker_segments[speaker].append({
                'start': start,
                'end': end
            })
        
        self.update_status(f"Found {len(speaker_segments)} speakers")
        return speaker_segments

    def preprocess_audio(self, input_path: str, duration_ms: int = None) -> str:
        """Preprocess audio file: trim (if specified), normalize, convert to mono WAV, and apply noise reduction."""
        audio = AudioSegment.from_file(input_path)
        audio_duration_ms = len(audio)
        if duration_ms and duration_ms > 0:
            self.update_status(f"Preprocessing audio for {duration_ms/1000:.1f} seconds (out of {audio_duration_ms/1000:.1f} seconds total)...")
            audio = audio[:duration_ms]
        else:
            self.update_status(f"Preprocessing full audio ({audio_duration_ms/1000:.1f} seconds)...")
        audio = audio + 10
        audio = audio.set_channels(1)
        audio = audio.set_frame_rate(16000)
        
        temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp')
        os.makedirs(temp_dir, exist_ok=True)
        output_path = os.path.join(temp_dir, 'preprocessed_audio.wav')
        audio.export(output_path, format='wav')
        
        sample_rate, data = wavfile.read(output_path)
        def high_pass_filter(data, cutoff=150, fs=44100, order=5):
            nyq = 0.5 * fs
            normal_cutoff = cutoff / nyq
            b, a = butter(order, normal_cutoff, btype='high', analog=False)
            return lfilter(b, a, data)
        filtered_data = high_pass_filter(data, fs=sample_rate)
        sf.write(output_path, filtered_data, sample_rate)
        
        self.update_status(f"Preprocessed audio saved to: {output_path}")
        return output_path

    def format_time(self, seconds: float) -> str:
        """Convert seconds to HH:MM:SS format."""
        return str(datetime.fromtimestamp(seconds).strftime("%H:%M:%S"))

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

if __name__ == "__main__":
    print("Starting audio transcription program with GUI...")
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.empty_cache()
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        print("CUDA is not available. Using CPU.")
    
    root = tk.Tk()
    app = TranscriptionApp(root)
    root.mainloop()