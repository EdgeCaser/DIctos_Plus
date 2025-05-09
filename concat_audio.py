import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from pydub import AudioSegment
import os
import threading

class AudioConcatGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Audio File Concatenator")
        self.root.geometry("600x400")
        
        # File A selection
        self.fileA_frame = ttk.LabelFrame(root, text="First Audio File (A)", padding="5")
        self.fileA_frame.pack(fill="x", padx=5, pady=5)
        
        self.fileA_path = tk.StringVar()
        self.fileA_entry = ttk.Entry(self.fileA_frame, textvariable=self.fileA_path, width=50)
        self.fileA_entry.pack(side="left", padx=5)
        
        self.browseA_btn = ttk.Button(self.fileA_frame, text="Browse", command=lambda: self.browse_file('A'))
        self.browseA_btn.pack(side="left", padx=5)
        
        self.durationA_var = tk.StringVar(value="Duration: --:--:--")
        self.durationA_label = ttk.Label(self.fileA_frame, textvariable=self.durationA_var)
        self.durationA_label.pack(side="left", padx=5)
        
        # File B selection
        self.fileB_frame = ttk.LabelFrame(root, text="Second Audio File (B)", padding="5")
        self.fileB_frame.pack(fill="x", padx=5, pady=5)
        
        self.fileB_path = tk.StringVar()
        self.fileB_entry = ttk.Entry(self.fileB_frame, textvariable=self.fileB_path, width=50)
        self.fileB_entry.pack(side="left", padx=5)
        
        self.browseB_btn = ttk.Button(self.fileB_frame, text="Browse", command=lambda: self.browse_file('B'))
        self.browseB_btn.pack(side="left", padx=5)
        
        self.durationB_var = tk.StringVar(value="Duration: --:--:--")
        self.durationB_label = ttk.Label(self.fileB_frame, textvariable=self.durationB_var)
        self.durationB_label.pack(side="left", padx=5)
        
        # Total duration
        self.total_frame = ttk.Frame(root)
        self.total_frame.pack(fill="x", padx=5, pady=5)
        self.total_duration_var = tk.StringVar(value="Total Duration: --:--:--")
        self.total_duration_label = ttk.Label(self.total_frame, textvariable=self.total_duration_var)
        self.total_duration_label.pack(side="left", padx=5)
        
        # Progress bar
        self.progress_frame = ttk.Frame(root)
        self.progress_frame.pack(fill="x", padx=5, pady=5)
        self.progress_bar = ttk.Progressbar(self.progress_frame, mode='determinate')
        self.progress_bar.pack(fill="x", padx=5)
        
        # Concatenate button
        self.concat_btn = ttk.Button(root, text="Concatenate Files", command=self.start_concat)
        self.concat_btn.pack(pady=20)
        
        # Progress
        self.progress_var = tk.StringVar()
        self.progress_label = ttk.Label(root, textvariable=self.progress_var)
        self.progress_label.pack(pady=5)
        
        # Processing flag
        self.processing = False
        
        # Store durations
        self.durationA_ms = 0
        self.durationB_ms = 0

    def milliseconds_to_time_str(self, ms):
        """Convert milliseconds to HH:MM:SS format"""
        total_seconds = int(ms / 1000)
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def load_audio_duration(self, file_path, file_label):
        """Load audio file and get its duration"""
        try:
            audio = AudioSegment.from_file(file_path)
            duration_ms = len(audio)
            duration_str = self.milliseconds_to_time_str(duration_ms)
            
            if file_label == 'A':
                self.durationA_var.set(f"Duration: {duration_str}")
                self.durationA_ms = duration_ms
            else:
                self.durationB_var.set(f"Duration: {duration_str}")
                self.durationB_ms = duration_ms
            
            # Update total duration
            total_ms = self.durationA_ms + self.durationB_ms
            total_str = self.milliseconds_to_time_str(total_ms)
            self.total_duration_var.set(f"Total Duration: {total_str}")
            
            return True
            
        except Exception as e:
            messagebox.showerror("Error", f"Could not load audio file: {str(e)}")
            return False

    def browse_file(self, file_label):
        file_path = filedialog.askopenfilename(
            filetypes=[("Audio Files", "*.mp3 *.wav *.m4a *.ogg")]
        )
        if file_path:
            if file_label == 'A':
                self.fileA_path.set(file_path)
            else:
                self.fileB_path.set(file_path)
            self.load_audio_duration(file_path, file_label)

    def set_progress(self, message, progress):
        self.progress_var.set(message)
        self.progress_bar['value'] = progress
        self.root.update_idletasks()

    def start_concat(self):
        if self.processing:
            return
        
        self.processing = True
        self.concat_btn.config(state='disabled')
        self.progress_bar['value'] = 0
        
        # Start processing thread
        thread = threading.Thread(target=self.concat_audio)
        thread.daemon = True
        thread.start()

    def update_ui(self, message, error=False):
        self.progress_var.set(message)
        if error:
            messagebox.showerror("Error", message)
        self.processing = False
        self.concat_btn.config(state='normal')
        self.progress_bar['value'] = 0

    def concat_audio(self):
        try:
            # Validate input files
            fileA_path = self.fileA_path.get()
            fileB_path = self.fileB_path.get()
            
            if not fileA_path or not fileB_path:
                self.root.after(0, lambda: self.update_ui("Please select both audio files", True))
                return

            # Load the audio files
            self.root.after(0, lambda: self.set_progress("Loading first audio file...", 20))
            audioA = AudioSegment.from_file(fileA_path)
            
            self.root.after(0, lambda: self.set_progress("Loading second audio file...", 40))
            audioB = AudioSegment.from_file(fileB_path)

            # Get output path
            self.root.after(0, lambda: self.set_progress("Select where to save...", 60))
            output_path = filedialog.asksaveasfilename(
                initialfile=f"combined_audio.mp3",
                defaultextension=".mp3",
                filetypes=[("MP3 Files", "*.mp3")]
            )
            
            if not output_path:
                self.root.after(0, lambda: self.update_ui("Operation cancelled"))
                return

            # Concatenate
            self.root.after(0, lambda: self.set_progress("Concatenating audio files...", 80))
            combined = audioA + audioB

            # Export
            self.root.after(0, lambda: self.set_progress("Saving combined audio...", 90))
            combined.export(output_path, format="mp3")
            
            self.root.after(0, lambda: self.set_progress("Done! File saved successfully.", 100))
            self.root.after(0, lambda: messagebox.showinfo("Success", "Audio files combined successfully!"))
            self.root.after(100, lambda: self.update_ui("Ready"))
            
        except Exception as e:
            self.root.after(0, lambda: self.update_ui(f"An error occurred: {str(e)}", True))

if __name__ == "__main__":
    root = tk.Tk()
    app = AudioConcatGUI(root)
    root.mainloop()
