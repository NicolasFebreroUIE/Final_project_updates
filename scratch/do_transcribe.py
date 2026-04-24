"""
Direct transcription of both videos using Whisper.
Bypasses Whisper's internal ffmpeg call by loading audio manually.
"""
import os
import sys
import json
import subprocess
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import imageio_ffmpeg
ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
print(f"[OK] ffmpeg: {ffmpeg_exe}")

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "transcripts")
os.makedirs(OUTPUT_DIR, exist_ok=True)

VIDEOS = {
    "suspect": os.path.join(PROJECT_ROOT, "data", "videos", "suspect_video.mp4"),
    "witness": os.path.join(PROJECT_ROOT, "data", "videos", "witness_video.mp4"),
}

def load_audio_manual(video_path):
    """Extract audio as numpy float32 array at 16kHz mono, using imageio_ffmpeg."""
    cmd = [
        ffmpeg_exe,
        "-nostdin",
        "-threads", "0",
        "-i", video_path,
        "-f", "s16le",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-"
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        print(f"  [FFMPEG STDERR] {result.stderr.decode('utf-8', errors='replace')[-500:]}")
        raise RuntimeError("ffmpeg failed")
    
    audio = np.frombuffer(result.stdout, np.int16).flatten().astype(np.float32) / 32768.0
    print(f"  [OK] Audio loaded: {len(audio)} samples ({len(audio)/16000:.1f}s)")
    return audio

def main():
    for name, path in VIDEOS.items():
        if not os.path.exists(path):
            print(f"[ERROR] Video not found: {path}")
            return
        size_mb = os.path.getsize(path) / (1024*1024)
        print(f"[OK] {name}: {path} ({size_mb:.1f} MB)")

    import whisper
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[LOADING] Whisper 'tiny' on {device}...")
    model = whisper.load_model("tiny", device=device)
    print("[OK] Model loaded.\n")

    for name, path in VIDEOS.items():
        print(f"[TRANSCRIBING] {name}...")
        
        # Load audio as numpy array (bypasses Whisper's internal ffmpeg call)
        audio = load_audio_manual(path)
        
        # Whisper expects a numpy array or a path. We pass the array directly.
        result = model.transcribe(audio, language="en", fp16=(device == "cuda"))
        
        segments = result.get("segments", [])
        full_text = result.get("text", "")
        
        print(f"  Full text: {full_text}")
        print(f"  Segments ({len(segments)}):")
        for seg in segments:
            print(f"    [{seg['start']:.1f}s - {seg['end']:.1f}s] {seg['text'].strip()}")
        
        out_path = os.path.join(OUTPUT_DIR, f"{name}_video.mp4.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(segments, f, indent=2, ensure_ascii=False)
        print(f"  [SAVED] {out_path}\n")

    print("[DONE] All transcriptions complete.")

if __name__ == "__main__":
    main()
