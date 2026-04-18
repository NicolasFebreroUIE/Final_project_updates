import mimetypes
mimetypes.add_type('video/mp4', '.mp4')
mimetypes.add_type('video/quicktime', '.mov')

import sys
import os

# Append parent dir so we can import models and phases
import os
try:
    import imageio_ffmpeg
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    ffmpeg_dir = os.path.dirname(ffmpeg_exe)
    if os.path.exists(ffmpeg_dir):
        os.environ["PATH"] += os.pathsep + ffmpeg_dir
except ImportError:
    pass

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_socketio import SocketIO
from flask_cors import CORS
import json

app = Flask(__name__)
app.config['SECRET_KEY'] = 'algorithmic-impartiality-paradox'
CORS(app)

# Use threading mode for maximum compatibility with ML libraries on Windows
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading', max_http_buffer_size=10*1024*1024)

# Import socketio events (we will update this file next)
from socketio_events import register_socketio_events
register_socketio_events(socketio)

@app.route('/')
def boot_screen():
    return render_template('index.html')

@app.route('/case')
def case_page():
    return render_template('case.html')

@app.route('/training')
def training_page():
    return render_template('training.html')

@app.route('/evidence')
def evidence_page():
    return render_template('evidence.html')

@app.route('/arguments')
def arguments_page():
    return render_template('arguments.html')

@app.route('/judge')
def judge_page():
    path = os.path.join(PROJECT_ROOT, "outputs", "phase1_results.json")
    scores = {}
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            scores = data.get("variant_scores", data.get("scores", {}))
    return render_template('judge.html', scores=scores)

@app.route('/avatar-verdict')
def avatar_page():
    return render_template('avatar_verdict.html')

@app.route('/videos')
def videos_page():
    return render_template('videos.html')

# API Endpoints
@app.route('/api/case-data')
def get_case_data():
    desc_path = os.path.join(PROJECT_ROOT, "case_description.md")
    with open(desc_path, 'r', encoding='utf-8') as f:
        content = f.read()
    return jsonify({"description": content})

@app.route('/api/phase1-results')
def get_phase1_results():
    path = os.path.join(PROJECT_ROOT, "outputs", "phase1_results.json")
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return jsonify(json.load(f))
    return jsonify({"error": "Results not found"}), 404

@app.route('/api/evidence-image')
def get_evidence_image():
    # Priority: check outputs for annotated version (the one with the green box)
    output_path = os.path.join(PROJECT_ROOT, 'outputs')
    if os.path.exists(os.path.join(output_path, 'physical_evidence_annotated.png')):
        return send_from_directory(output_path, 'physical_evidence_annotated.png')
    
    # Secondary: check the static forensics folder for the base image
    static_path = os.path.join(PROJECT_ROOT, 'web', 'static', 'data', 'forensics')
    if os.path.exists(os.path.join(static_path, 'weapon_image.jpg')):
        return send_from_directory(static_path, 'weapon_image.jpg')
    
    return jsonify({"error": "No image found"}), 404

@app.route('/api/evidence-report')
def get_evidence_report():
    path = os.path.join(PROJECT_ROOT, "phase2_cv/reports/physical_evidence_report.json")
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return jsonify(json.load(f))
    return jsonify({"error": "Report not found"}), 404

# Serve static outputs (images, etc)
@app.route('/outputs/<path:filename>')
def serve_outputs(filename):
    return send_from_directory(os.path.join(PROJECT_ROOT, 'outputs'), filename)

# Serve audio
@app.route('/static/audio/<path:filename>')
def serve_audio(filename):
    return send_from_directory(os.path.join(PROJECT_ROOT, 'web', 'static', 'audio'), filename)

# Serve videos from data/videos
@app.route('/data/videos/<path:filename>')
def serve_videos(filename):
    return send_from_directory(os.path.join(PROJECT_ROOT, 'data', 'videos'), filename)

def perform_wipe():
    """Internal helper to wipe all session artifacts."""
    paths_to_clear = [
        os.path.join(PROJECT_ROOT, "outputs", "phase1_results.json"),
        os.path.join(PROJECT_ROOT, "outputs", "phase4_results.json"),
        os.path.join(PROJECT_ROOT, "outputs", "physical_evidence_annotated.png"),
        os.path.join(PROJECT_ROOT, "phase2_cv", "reports", "physical_evidence_report.json"),
        os.path.join(PROJECT_ROOT, "phase2_cv", "reports", "suspect_video_report.json"),
        os.path.join(PROJECT_ROOT, "phase2_cv", "reports", "witness_video_report.json"),
        os.path.join(PROJECT_ROOT, "web", "static", "audio", "verdict.mp3")
    ]
    deleted = []
    for p in paths_to_clear:
        if os.path.exists(p):
            try:
                os.remove(p)
                deleted.append(os.path.basename(p))
            except Exception as e:
                print(f"Error deleting {p}: {e}")
    return deleted

@app.route('/api/reset-session', methods=['POST'])
def reset_session():
    """Wipes all forensic and video reports for a fresh session."""
    deleted = perform_wipe()
    print(f"[RESET] Cleared artifacts via API: {deleted}")
    return jsonify({"status": "success", "cleared": deleted})

if __name__ == '__main__':
    print("ALGORITHMIC IMPARTIALITY PARADOX SYSTEM")
    # AUTO-WIPE ON MANUAL STARTUP (Prevents wipe on debug reloads)
    if os.environ.get('WERKZEUG_RUN_MAIN') != 'true':
        print("[STARTUP] Performing fresh session wipe...")
        perform_wipe()
    
    print("Starting server on http://localhost:5001")
    socketio.run(app, host='0.0.0.0', port=5001, debug=True, allow_unsafe_werkzeug=True)
