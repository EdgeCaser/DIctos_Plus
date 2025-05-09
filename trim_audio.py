import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from pydub import AudioSegment
import os
import re
import threading
import time

class AudioTrimmerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Audio Trimmer")
        self.root.geometry("500x400")
        
        # File selection
        self.file_frame = ttk.LabelFrame(root, text="Audio File", padding="5")
        self.file_frame.pack(fill="x", padx=5, pady=5)
        
        self.file_path = tk.StringVar()
        self.file_entry = ttk.Entry(self.file_frame, textvariable=self.file_path, width=50)
        self.file_entry.pack(side="left", padx=5)
        
        self.browse_btn = ttk.Button(self.file_frame, text="Browse", command=self.browse_file)
        self.browse_btn.pack(side="left", padx=5)
        
        # Duration label
        self.duration_var = tk.StringVar(value="Duration: --:--:--")
        self.duration_label = ttk.Label(self.file_frame, textvariable=self.duration_var)
        self.duration_label.pack(side="left", padx=5)
        
        # Time inputs
        self.time_frame = ttk.LabelFrame(root, text="Time Range", padding="5")
        self.time_frame.pack(fill="x", padx=5, pady=5)
        
        # Start time
        self.start_frame = ttk.Frame(self.time_frame)
        self.start_frame.pack(fill="x", pady=5)
        ttk.Label(self.start_frame, text="Start Time (HH:MM:SS):").pack(side="left", padx=5)
        self.start_time = ttk.Entry(self.start_frame, width=10)
        self.start_time.pack(side="left", padx=5)
        self.start_time.insert(0, "00:00:00")
        
        # End time
        self.end_frame = ttk.Frame(self.time_frame)
        self.end_frame.pack(fill="x", pady=5)
        ttk.Label(self.end_frame, text="End Time (HH:MM:SS):  ").pack(side="left", padx=5)
        self.end_time = ttk.Entry(self.end_frame, width=10)
        self.end_time.pack(side="left", padx=5)
        self.end_time.insert(0, "00:00:00")
        
        # Progress bar
        self.progress_frame = ttk.Frame(root)
        self.progress_frame.pack(fill="x", padx=5, pady=5)
        self.progress_bar = ttk.Progressbar(self.progress_frame, mode='determinate')
        self.progress_bar.pack(fill="x", padx=5)
        
        # Trim button
        self.trim_btn = ttk.Button(root, text="Trim Audio", command=self.start_trim)
        self.trim_btn.pack(pady=20)
        
        # Progress
        self.progress_var = tk.StringVar()
        self.progress_label = ttk.Label(root, textvariable=self.progress_var)
        self.progress_label.pack(pady=5)
        
        # Processing flag
        self.processing = False
        
        # Store audio duration
        self.audio_duration_ms = 0

    def milliseconds_to_time_str(self, ms):
        """Convert milliseconds to HH:MM:SS format"""
        total_seconds = int(ms / 1000)
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def load_audio_duration(self, file_path):
        """Load audio file and get its duration"""
        try:
            self.progress_var.set("Loading audio file...")
            self.progress_bar['value'] = 10
            self.root.update_idletasks()
            
            audio = AudioSegment.from_file(file_path)
            self.audio_duration_ms = len(audio)
            duration_str = self.milliseconds_to_time_str(self.audio_duration_ms)
            
            self.duration_var.set(f"Duration: {duration_str}")
            self.end_time.delete(0, tk.END)
            self.end_time.insert(0, duration_str)
            
            self.progress_var.set("Ready")
            self.progress_bar['value'] = 0
            return True
            
        except Exception as e:
            self.progress_var.set("Error loading audio file")
            messagebox.showerror("Error", f"Could not load audio file: {str(e)}")
            return False

    def browse_file(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Audio Files", "*.mp3 *.wav *.m4a *.ogg")]
        )
        if file_path:
            self.file_path.set(file_path)
            self.load_audio_duration(file_path)

    def time_to_milliseconds(self, time_str):
        pattern = re.compile(r'^(\d{1,2}):(\d{1,2}):(\d{1,2})$')
        match = pattern.match(time_str)
        if not match:
            raise ValueError("Invalid time format. Use HH:MM:SS")
        
        hours, minutes, seconds = map(int, match.groups())
        return (hours * 3600 + minutes * 60 + seconds) * 1000

    def set_progress(self, message, progress):
        self.progress_var.set(message)
        self.progress_bar['value'] = progress
        self.root.update_idletasks()

    def start_trim(self):
        if self.processing:
            return
        
        self.processing = True
        self.trim_btn.config(state='disabled')
        self.progress_bar['value'] = 0
        
        # Start processing thread
        thread = threading.Thread(target=self.trim_audio)
        thread.daemon = True
        thread.start()

    def update_ui(self, message, error=False):
        self.progress_var.set(message)
        if error:
            messagebox.showerror("Error", message)
        self.processing = False
        self.trim_btn.config(state='normal')
        self.progress_bar['value'] = 0

    def trim_audio(self):
        try:
            # Validate input file
            input_path = self.file_path.get()
            if not input_path:
                self.root.after(0, lambda: self.update_ui("Please select an audio file", True))
                return

            # Convert times to milliseconds
            start_ms = self.time_to_milliseconds(self.start_time.get())
            end_ms = self.time_to_milliseconds(self.end_time.get())

            # Load the audio
            self.root.after(0, lambda: self.set_progress("Loading audio file...", 10))
            audio = AudioSegment.from_file(input_path)
            time.sleep(0.5)  # Small delay to show progress

            # Validate times
            if end_ms > len(audio):
                end_ms = len(audio)
            if start_ms >= end_ms:
                self.root.after(0, lambda: self.update_ui("Start time must be before end time", True))
                return

            # Get output path
            self.root.after(0, lambda: self.set_progress("Select where to save...", 30))
            output_path = filedialog.asksaveasfilename(
                initialfile=f"{os.path.splitext(os.path.basename(input_path))[0]}_trimmed.mp3",
                defaultextension=".mp3",
                filetypes=[("MP3 Files", "*.mp3")]
            )
            
            if not output_path:
                self.root.after(0, lambda: self.update_ui("Operation cancelled"))
                return

            # Trim the audio
            self.root.after(0, lambda: self.set_progress("Trimming audio...", 60))
            time.sleep(0.5)  # Small delay to show progress
            trimmed_audio = audio[start_ms:end_ms]

            # Export
            self.root.after(0, lambda: self.set_progress("Saving trimmed audio...", 80))
            time.sleep(0.5)  # Small delay to show progress
            trimmed_audio.export(output_path, format="mp3")
            
            self.root.after(0, lambda: self.set_progress("Done! File saved successfully.", 100))
            time.sleep(0.5)  # Small delay to show completion
            self.root.after(0, lambda: messagebox.showinfo("Success", "Audio trimmed successfully!"))
            self.root.after(100, lambda: self.update_ui("Ready"))
            
        except ValueError as e:
            self.root.after(0, lambda: self.update_ui(str(e), True))
        except Exception as e:
            self.root.after(0, lambda: self.update_ui(f"An error occurred: {str(e)}", True))

if __name__ == "__main__":
    root = tk.Tk()
    app = AudioTrimmerGUI(root)
    root.mainloop()
