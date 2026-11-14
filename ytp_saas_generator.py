import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import librosa
import soundfile as sf
from scipy.signal import fftconvolve
import threading
import subprocess
import os
import sys

if os.name == 'nt':
    import winsound

if getattr(sys, 'frozen', False):
    app_dir = os.path.dirname(sys.executable)
    bundle_dir = getattr(sys, '_MEIPASS', app_dir)
    os.environ["PATH"] = app_dir + os.pathsep + bundle_dir + os.pathsep + os.environ.get("PATH", "")
else:
    app_dir = os.path.dirname(os.path.abspath(__file__))
    os.environ["PATH"] = app_dir + os.pathsep + os.environ.get("PATH", "")

def apply_reverb(audio, sr, wet=0.35, decay=0.4):
    if wet <= 0:
        return audio
    reverb_len = int(sr * decay)
    t = np.linspace(0, decay, reverb_len)
    ir = np.random.randn(reverb_len) * np.exp(-5 * t)
    ir = ir / np.max(np.abs(ir))
    
    wet_signal = fftconvolve(audio, ir, mode='full')[:len(audio)]
    max_val = np.max(np.abs(wet_signal))
    if max_val > 0:
        wet_signal = wet_signal / max_val * np.max(np.abs(audio))
        
    return (1 - wet) * audio + wet * wet_signal

def process_audio(input_path, output_path, slice_ms, reverb_wet, chance, status_callback, done_callback):
    try:
        status_callback("Lade Audiodatei...")
        y, sr = librosa.load(input_path, sr=44100, mono=True)
        
        status_callback("Suche Silben / Vokale...")
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr, backtrack=True)
        onset_samples = librosa.frames_to_samples(onset_frames)
        
        slice_len = int((slice_ms / 1000.0) * sr)
        output_chunks = []
        last_idx = 0
        
        status_callback("Erzeuge Saas/Soos-Effekte...")
        for idx in onset_samples:
            if idx < last_idx:
                continue
                
            output_chunks.append(y[last_idx:idx])
            
            if np.random.rand() < chance and (idx + slice_len) < len(y):
                syllable = y[idx : idx + slice_len]
                reversed_syllable = syllable[::-1]
                
                fade_len = min(int(0.005 * sr), len(syllable) // 2)
                fade_in = np.linspace(0, 1, fade_len)
                fade_out = np.linspace(1, 0, fade_len)
                
                syllable_faded = syllable.copy()
                rev_faded = reversed_syllable.copy()
                syllable_faded[-fade_len:] *= fade_out
                rev_faded[:fade_len] *= fade_in
                
                saas_effect = np.concatenate([syllable_faded, rev_faded])
                output_chunks.append(saas_effect)
                last_idx = idx + slice_len
            else:
                last_idx = idx

        output_chunks.append(y[last_idx:])
        final_audio = np.concatenate(output_chunks)
        
        if reverb_wet > 0:
            status_callback("Wende Reverb (Hall) an...")
            final_audio = apply_reverb(final_audio, sr, wet=reverb_wet)
            
        max_amp = np.max(np.abs(final_audio))
        if max_amp > 0:
            final_audio = final_audio / max_amp * 0.95
            
        status_callback("Speichere Ausgabedatei...")
        
        if output_path.lower().endswith(".mp3"):
            temp_wav = output_path + "_temp.wav"
            sf.write(temp_wav, final_audio, sr)
            
            startupinfo = None
            if os.name == 'nt':
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                
            subprocess.run(
                ["ffmpeg", "-y", "-i", temp_wav, "-b:a", "192k", output_path],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                startupinfo=startupinfo, check=True
            )
            if os.path.exists(temp_wav):
                os.remove(temp_wav)
        else:
            sf.write(output_path, final_audio, sr)
            
        status_callback("Fertig! Klicke auf 'Vorschau' zum Anhören.")
        done_callback(output_path)
        
    except Exception as e:
        status_callback("Fehler aufgetreten!")
        messagebox.showerror("Fehler", str(e))

class SaasApp:
    def __init__(self, root):
        self.root = root
        self.root.title("YTP Saas/Soos Generator")
        self.root.geometry("500x480")
        self.root.resizable(False, False)
        
        self.input_file = tk.StringVar()
        self.output_file = tk.StringVar()
        self.slice_ms = tk.IntVar(value=220)
        self.reverb_val = tk.DoubleVar(value=0.3)
        self.chance_val = tk.DoubleVar(value=0.6)
        self.status_text = tk.StringVar(value="Bereit.")
        self.last_generated_file = None
        self.preview_wav_path = os.path.join(os.environ.get("TEMP", "."), "ytp_saas_preview.wav")
        
        tk.Label(root, text="1. Sprachdatei wählen (WAV, MP3, OPUS, M4A, OGG):").pack(anchor="w", padx=15, pady=(10, 2))
        frame_in = tk.Frame(root)
        frame_in.pack(fill="x", padx=15)
        tk.Entry(frame_in, textvariable=self.input_file, width=42).pack(side="left")
        tk.Button(frame_in, text="Suchen...", command=self.browse_input).pack(side="right")
        
        tk.Label(root, text="2. Zielpfad & Format wählen (MP3 oder WAV):").pack(anchor="w", padx=15, pady=(10, 2))
        frame_out = tk.Frame(root)
        frame_out.pack(fill="x", padx=15)
        tk.Entry(frame_out, textvariable=self.output_file, width=42).pack(side="left")
        tk.Button(frame_out, text="Ziel ändern...", command=self.browse_output).pack(side="right")
        
        tk.Label(root, text="Silben-Länge für Saas/Soos (Millisekunden):").pack(anchor="w", padx=15, pady=(12, 0))
        tk.Scale(root, from_=100, to=500, orient="horizontal", variable=self.slice_ms).pack(fill="x", padx=15)
        
        tk.Label(root, text="Reverb / Hall-Intensität (0 = aus, 1 = stark):").pack(anchor="w", padx=15, pady=(5, 0))
        tk.Scale(root, from_=0.0, to=0.8, resolution=0.05, orient="horizontal", variable=self.reverb_val).pack(fill="x", padx=15)
        
        tk.Label(root, text="Saas-Häufigkeit (Wahrscheinlichkeit pro Silbe):").pack(anchor="w", padx=15, pady=(5, 0))
        tk.Scale(root, from_=0.1, to=1.0, resolution=0.1, orient="horizontal", variable=self.chance_val).pack(fill="x", padx=15)
        
        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=15)
        
        tk.Button(btn_frame, text="SAAS / SOOS GENERIEREN", bg="#4CAF50", fg="white", font=("Arial", 10, "bold"),
                  command=self.start_processing).pack(side="left", padx=5, ipadx=8, ipady=4)
        
        self.btn_play = tk.Button(btn_frame, text="Vorschau", bg="#2196F3", fg="white", font=("Arial", 10, "bold"),
                                  state="disabled", command=self.play_preview)
        self.btn_play.pack(side="left", padx=5, ipadx=8, ipady=4)
        
        self.btn_stop = tk.Button(btn_frame, text="Stop", bg="#f44336", fg="white", font=("Arial", 10, "bold"),
                                  state="disabled", command=self.stop_preview)
        self.btn_stop.pack(side="left", padx=5, ipadx=6, ipady=4)
        
        tk.Label(root, textvariable=self.status_text, fg="gray").pack()
        
        tk.Label(root, text="made by hurius", font=("Arial", 8, "italic"), fg="#888888").pack(side="bottom", anchor="e", padx=12, pady=6)
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def browse_input(self):
        path = filedialog.askopenfilename(filetypes=[("Audio Dateien", "*.wav *.mp3 *.opus *.m4a *.ogg *.flac")])
        if path:
            self.input_file.set(path)
            base, _ = os.path.splitext(path)
            self.output_file.set(base + "_SAAS.mp3")

    def browse_output(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".mp3",
            filetypes=[("MP3 Audio", "*.mp3"), ("WAV Audio", "*.wav")]
        )
        if path:
            self.output_file.set(path)

    def update_status(self, text):
        self.status_text.set(text)
        self.root.update_idletasks()

    def on_generation_done(self, output_path):
        self.last_generated_file = output_path
        self.btn_play.config(state="normal")
        self.btn_stop.config(state="normal")

    def start_processing(self):
        self.stop_preview()
        in_p = self.input_file.get()
        out_p = self.output_file.get()
        if not in_p or not out_p:
            messagebox.showwarning("Fehlende Angaben", "Bitte Eingabe- und Ausgabedatei auswählen!")
            return
        
        self.btn_play.config(state="disabled")
        self.btn_stop.config(state="disabled")
        
        t = threading.Thread(target=process_audio, args=(
            in_p, out_p, self.slice_ms.get(), self.reverb_val.get(), self.chance_val.get(),
            self.update_status, self.on_generation_done
        ))
        t.start()

    def play_preview(self):
        if not self.last_generated_file or not os.path.exists(self.last_generated_file):
            return
            
        self.stop_preview()
        try:
            if os.name == 'nt':
                if self.last_generated_file.lower().endswith(".mp3"):
                    startupinfo = subprocess.STARTUPINFO()
                    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                    subprocess.run(
                        ["ffmpeg", "-y", "-i", self.last_generated_file, self.preview_wav_path],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                        startupinfo=startupinfo, check=True
                    )
                    play_target = self.preview_wav_path
                else:
                    play_target = self.last_generated_file
                    
                winsound.PlaySound(play_target, winsound.SND_FILENAME | winsound.SND_ASYNC)
                self.update_status("Spiele Vorschau ab...")
            else:
                subprocess.Popen(["aplay", self.last_generated_file], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            messagebox.showerror("Vorschau-Fehler", f"Konnte Datei nicht abspielen:\n{e}")

    def stop_preview(self):
        if os.name == 'nt':
            winsound.PlaySound(None, winsound.SND_PURGE)
        self.update_status("Wiedergabe gestoppt.")

    def on_close(self):
        self.stop_preview()
        if os.path.exists(self.preview_wav_path):
            try:
                os.remove(self.preview_wav_path)
            except OSError:
                pass
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = SaasApp(root)
    root.mainloop()
