import os
import whisper
import json
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIDEOS = ["suspect_video.mp4", "witness_video.mp4"]
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "transcripts")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def pre_transcribe():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Whisper model on {device}...")
    model = whisper.load_model("base", device=device)
    
    for v in VIDEOS:
        video_path = os.path.join(PROJECT_ROOT, "data", "videos", v)
        output_path = os.path.join(OUTPUT_DIR, f"{v}.json")
        
        if os.path.exists(output_path):
            print(f"Transcript already exists for {v}")
            continue
            
        if not os.path.exists(video_path):
            print(f"Video not found: {video_path}")
            continue
            
        print(f"Transcribing {v}...")
        result = model.transcribe(video_path, language="en")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result.get('segments', []), f, indent=4)
        print(f"Done: {output_path}")

if __name__ == "__main__":
    pre_transcribe()
