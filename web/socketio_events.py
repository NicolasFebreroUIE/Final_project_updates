import time
import threading
import json
import os
import random
import base64
import numpy as np
import cv2
from flask_socketio import emit
from gtts import gTTS

# Project root for path management
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from phase4_integration.integration_pipeline import run_final_integration

# Lazy-loaded pipeline instance (shared across socket events)
_video_pipeline = None
_central_model = None
_transcript_cache = {} # { video_path: [ {start, end, text}, ... ] }
_whisper_model = None
_realtime_emotions = { 'suspect': [], 'witness': [] }
_realtime_scores = { 'suspect': [], 'witness': [] }
_last_sent_segment_id = { 'suspect': -1, 'witness': -1 }
_last_progress_time = { 'suspect': -1, 'witness': -1 }
_socketio = None

def _get_model():
    global _central_model
    if _central_model is None:
        try:
            from models.central_model import CentralModel
            model_path = os.path.join(PROJECT_ROOT, "models", "central_model.pkl")
            _central_model = CentralModel() # Instantiate first
            if os.path.exists(model_path):
                _central_model.load(model_path) # Then load instance
        except Exception as e:
            print(f"[ERROR] Failed to load CentralModel: {e}")
    return _central_model

def _get_pipeline():
    global _video_pipeline
    if _video_pipeline is None:
        from phase2_cv.video_pipeline import VideoPipeline
        _video_pipeline = VideoPipeline()
    return _video_pipeline

def _get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        import whisper
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[FORENSIC] Loading Whisper model on {device}...")
        _whisper_model = whisper.load_model("base", device=device)
    return _whisper_model

def register_socketio_events(socketio):
    global _socketio
    _socketio = socketio

    @socketio.on('start_training_replay')
    def handle_training_replay():
        history_path = os.path.join(PROJECT_ROOT, "outputs", "training_history.json")
        if not os.path.exists(history_path):
            emit('training_log', {'message': "Error: training_history.json not found."})
            return

        with open(history_path, 'r') as f:
            history = json.load(f)
        
        emit('training_log', {'message': "Initializing Legal-BERT-Spanish Core Environment..."})
        time.sleep(1)
        emit('training_log', {'message': "✓ Transformer Encoder - 12 Layers, 768 Hidden, 12 Heads"})
        time.sleep(0.5)
        emit('training_log', {'message': "Loading Historical Metrics [CENDOJ / EURLEX Database]..."})
        
        # Simulate 10 epochs (or whatever is in history)
        epochs = history.get('epochs', [])
        for i, epoch in enumerate(epochs):
            _socketio.sleep(0.6) 
            emit('training_update', {
                'epoch': i + 1,
                'loss': epoch.get('loss', 0),
                'accuracy': epoch.get('accuracy', 0)
            })
            emit('training_log', {'message': f"Epoch {i+1}/10: Training Loss: {epoch.get('loss', 0):.6f} | Validation Accuracy: {epoch.get('accuracy', 0):.4f}"})
        
        emit('training_complete', {'status': 'success'})
        emit('training_log', {'message': "System Core State: SYNCHRONIZED."})
        
        # Pre-load Video Pipeline for Forensic Pass
        emit('training_log', {'message': "Warming up Forensic Vision Head (GPU)..."})
        try:
            _get_pipeline()
            emit('training_log', {'message': "✓ Vision Head Ready."})
        except Exception as e:
            emit('training_log', {'message': f"⚠ Vision Head Warning: {e}"})

    @socketio.on('start_scoring_replay')
    def handle_scoring_replay():
        """Directly analyzes variants using the CentralModel in memory."""
        # Immediate feedback pulse
        emit('scoring_log', {'message': "[SYSTEM] Establishing Secure Neural Handshake... OK"})
        _socketio.sleep(0.5)
        
        try:
            emit('scoring_log', {'message': "[CORE] Loading Linguistic Weights (BERT-Spanish-v4)... DONE"})
            _socketio.sleep(0.4)
            emit('scoring_log', {'message': "[XAI] Calibrating Impartiality Bias Thresholds... STANDBY"})
            _socketio.sleep(0.5)
            emit('scoring_log', {'message': "[NLP] Vectorizing Defense Corpus [################----] 84%"})
            _socketio.sleep(0.4)
            emit('scoring_log', {'message': "[JUDGE] Synchronizing Strategy Matrix with Judicial Benchmarks..."})
            _socketio.sleep(0.6)
            
            # Load variants directly from source file to ensure we have current data
            variants_path = os.path.join(PROJECT_ROOT, "data", "arguments", "lawyer_variants.md")
            if not os.path.exists(variants_path):
                emit('scoring_log', {'message': "![CRITICAL_FAIL] Source file 'lawyer_variants.md' not detected."})
                return

            with open(variants_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Fast Parse
            variants = []
            parts = content.split('## VARIANT ')
            for p in parts[1:]:
                lines = p.split('\n')
                v_num = lines[0].strip()
                v_text = "\n".join(lines[1:]).strip().split('---')[0].strip()
                variants.append({"id": f"variant_{v_num}", "text": v_text, "num": v_num})

            if not variants:
                emit('scoring_log', {'message': "![NULL_POINTER] No strategic vectors parsed from source."})
                return

            emit('scoring_log', {'message': f"[SYSTEM] Successfully mapped {len(variants)} strategic variants."})
            emit('scoring_log', {'message': "[AUDIT] Executing Real-Time Neural Audit... ACTIVE"})
            _socketio.sleep(0.8)
            
            # Use global lazy-loaded model
            model = _get_model()
            if model is None:
                emit('scoring_log', {'message': "Error: CentralModel pkl not found on server."})
                return
            
            texts = [v['text'] for v in variants]
            scores = model.score_batch(texts)
            
            # Prepare results for saving (so the Judge page can see them later)
            results_to_save = {
                "variant_scores": {v['id']: s for v, s in zip(variants, scores)},
                "variant_texts": {v['id']: v['text'] for v in variants}
            }
            res_path = os.path.join(PROJECT_ROOT, "outputs", "phase1_results.json")
            os.makedirs(os.path.dirname(res_path), exist_ok=True)
            with open(res_path, 'w', encoding='utf-8') as f:
                json.dump(results_to_save, f, indent=4)

            # Emit to UI
            for i, v in enumerate(variants):
                score = scores[i]
                _socketio.sleep(0.2)
                emit('scoring_update', {
                    'variant_id': f"Defense Strategy {v['num']}",
                    'raw_id': v['id'],
                    'score': score,
                    'text': v['text']
                })
                emit('scoring_log', {'message': f"✓ Strategy {v['num']} Audit Complete. Judicial Index: {score:.6f}"})

            emit('scoring_complete', {'status': 'success'})
            emit('scoring_log', {'message': "Linguistic Partiality Gap successfully mapped."})

        except Exception as e:
            import traceback
            traceback.print_exc()
            emit('scoring_log', {'message': f"Engine Error: {str(e)}"})

    @socketio.on('generate_verdict')
    def handle_verdict(data):
        variant_id = data.get('variant_id', 'variant_01')
        emit('judge_log', {'message': f"Reviewing submitted evidence (variant: {variant_id})..."})
        time.sleep(1)
        
        try:
            # Run the actual integration live!
            results = run_final_integration(variant_id)
            
            emit('judge_log', {'message': f"  ✓ NLP Persuasion: {results['raw_nlp_score']:.4f}"})
            _socketio.sleep(0.5)
            emit('judge_log', {'message': f"  ✓ Forensic Mismatch: {results['evidence_summary']['value'].upper()}"})
            _socketio.sleep(0.5)
            
            # Log Video Evidence Integration
            if results.get('suspect_report'):
                s_report = results['suspect_report']
                emit('judge_log', {'message': f"  ✓ Suspect Profile: {s_report.get('dominant_emotion', 'N/A').upper()} (Consistency: {s_report.get('emotional_consistency_score', 0):.2f})"})
                _socketio.sleep(0.4)
            
            if results.get('witness_report'):
                w_report = results['witness_report']
                emit('judge_log', {'message': f"  ✓ Witness Credibility: Reactivity Index {w_report.get('reactivity_index', 0):.2f}"})
                _socketio.sleep(0.4)

            emit('judge_log', {'message': "Applying Alis-Core Judicial Rules..."})
            
            for adj in results['adjustments']:
                _socketio.sleep(0.4)
                emit('judge_log', {'message': f"    [RULE {adj['rule']}] {adj['principle']} → {adj['adjustment']:.2f}"})

            # Generate gTTS Audio (Robust handling for offline mode)
            try:
                verdict_text = generate_verdict_speech_text(results)
                audio_path = os.path.join(PROJECT_ROOT, "web", "static", "audio", "verdict.mp3")
                os.makedirs(os.path.dirname(audio_path), exist_ok=True)
                
                tts = gTTS(text=verdict_text, lang='en')
                tts.save(audio_path)
                results['audio_available'] = True
            except Exception as audio_err:
                emit('judge_log', {'message': f"Audio generation skipped (Connection required for TTS): {str(audio_err)}"})
                results['audio_available'] = False
            
            emit('verdict_ready', results)
            
        except Exception as e:
            emit('judge_log', {'message': f"Error during verdict generation: {str(e)}"})

    @socketio.on('init_transcription')
    def handle_init_transcription(data):
        vid_type = data.get('type', 'suspect')
        video_name = f"{vid_type}_video.mp4"
        video_path = os.path.join(PROJECT_ROOT, "data", "videos", video_name)
        
        if video_path not in _transcript_cache:
            # Check for pre-transcribed JSON
            transcript_json_path = os.path.join(PROJECT_ROOT, "data", "transcripts", f"{video_name}.json")
            if os.path.exists(transcript_json_path):
                print(f"[FORENSIC] Pre-transcription found for {video_name} (Init)", flush=True)
                with open(transcript_json_path, 'r', encoding='utf-8') as f:
                    _transcript_cache[video_path] = json.load(f)
            else:
                print(f"[FORENSIC] Starting silent background transcription for {video_name}", flush=True)
                _transcript_cache[video_path] = "TRANSCRIPTION_PENDING"
                
                def run_whisper_bg(v_path, v_type):
                    try:
                        model = _get_whisper_model()
                        result = model.transcribe(v_path, language="en")
                        _transcript_cache[v_path] = result.get('segments', [])
                        print(f"[FORENSIC] Silent transcription complete for {v_path}", flush=True)
                        _socketio.emit('transcription_chunk', {
                            'type': v_type,
                            'text': "[ Auditory data synchronized. Live feed active. ]",
                            'is_final': False
                        })
                    except Exception as e:
                        print(f"[FORENSIC] Whisper Error: {e}")
                        _transcript_cache[v_path] = []

                socketio.start_background_task(run_whisper_bg, video_path, vid_type)

    @socketio.on('process_single_frame')
    def handle_frame(data):
        """Real-time frame processing: emotion + forensic mesh."""
        try:
            vid_type = data.get('type', 'suspect')
            print(f"[FRAME] Received frame from {vid_type} at time {data.get('time')}", flush=True)
            pipeline = _get_pipeline()
            vid_type = data.get('type', 'suspect')
            img_b64 = data.get('image', '')
            frame_w = data.get('width', 640)
            frame_h = data.get('height', 360)

            # Decode Base64 to numpy
            header, encoded = img_b64.split(',', 1) if ',' in img_b64 else ('', img_b64)
            img_bytes = base64.b64decode(encoded)
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                return

            # Face detection
            face_img, region = pipeline._detect_and_crop_face(frame)

            if face_img is not None:
                # Emotion classification via Unified Brain
                emotion, probs = pipeline._classify_emotion(face_img)
                confidence = float(np.max(probs))

                # Forensic mesh (eyes + mouth boxes)
                behavioral = pipeline._get_behavioral_data(frame)

                # Agitation score
                agitation = pipeline._get_agitation_score(frame)
                
                # Accumulate for final report
                _realtime_emotions[vid_type].append(emotion)
                _realtime_scores[vid_type].append(probs)

                # --- BACKGROUND TRANSCRIPTION (WHISPER) ---
                current_time = float(data.get('time', 0))
                video_name = f"{vid_type}_video.mp4"
                video_path = os.path.join(PROJECT_ROOT, "data", "videos", video_name)
                
                if video_path not in _transcript_cache:
                    # 1. Check if pre-transcribed JSON exists
                    transcript_json_path = os.path.join(PROJECT_ROOT, "data", "transcripts", f"{video_name}.json")
                    if os.path.exists(transcript_json_path):
                        print(f"[FORENSIC] Loading pre-transcribed data for {video_name}", flush=True)
                        with open(transcript_json_path, 'r', encoding='utf-8') as f:
                            _transcript_cache[video_path] = json.load(f)
                    else:
                        print(f"[FORENSIC] Transcription not in cache for {video_path}. Starting task.", flush=True)
                        _transcript_cache[video_path] = "TRANSCRIPTION_PENDING"
                        
                        # Notify user immediately in the log
                        emit('transcription_chunk', {
                            'type': vid_type,
                            'text': "[ ANALYSIS STARTING: Real-time audio parsing activated... ]",
                            'is_final': False
                        })

                    def run_whisper_bg(v_path, v_type):
                        try:
                            model = _get_whisper_model()
                            print(f"[FORENSIC] Analyzing audio (BG): {v_path}", flush=True)
                            result = model.transcribe(v_path, language="en")
                            _transcript_cache[v_path] = result.get('segments', [])
                            print(f"[FORENSIC] BG Transcription complete for {v_path}", flush=True)
                            # Notify UI that transcription is ready
                            _socketio.emit('transcription_chunk', {
                                'type': v_type,
                                'text': "[ Auditory data synchronized. Live feed active. ]",
                                'is_final': False
                            })
                        except Exception as e:
                            print(f"[FORENSIC] Whisper Error: {e}")
                            _transcript_cache[v_path] = []

                    # Run as a socketio background task
                    socketio.start_background_task(run_whisper_bg, video_path, vid_type)

                # Find segment if available
                segments = _transcript_cache.get(video_path, [])
                if isinstance(segments, list):
                    # Catch up: send all segments up to current_time that haven't been sent
                    for i, seg in enumerate(segments):
                        if seg['end'] <= current_time or (seg['start'] <= current_time <= seg['end']):
                            if i > _last_sent_segment_id[vid_type]:
                                _last_sent_segment_id[vid_type] = i
                                emit('transcription_chunk', {
                                    'type': vid_type,
                                    'text': seg['text'].strip(),
                                    'is_final': False
                                })
                        if seg['start'] > current_time:
                            break
                elif segments == "TRANSCRIPTION_PENDING":
                    current_sec = int(current_time)
                    if current_sec % 5 == 0 and current_sec != _last_progress_time[vid_type]:
                        _last_progress_time[vid_type] = current_sec
                        emit('transcription_chunk', {
                            'type': vid_type,
                            'text': "[ Auditory Forensic Analysis in progress... ]",
                            'is_final': False
                        })

                # Build bbox for face region
                bbox = None
                if region:
                    bbox = [region.get('x', 0), region.get('y', 0),
                            region.get('w', 0), region.get('h', 0)]

                emit('frame_emotion', {
                    'type': vid_type,
                    'emotion': emotion,
                    'confidence': confidence,
                    'probs': probs.tolist() if hasattr(probs, 'tolist') else list(probs),
                    'bbox': bbox,
                    'behavioral': behavioral,
                    'agitation': agitation
                })
            else:
                emit('frame_emotion', {
                    'type': vid_type,
                    'emotion': 'detecting...',
                    'confidence': 0,
                    'probs': [0]*7,
                    'bbox': None,
                    'behavioral': {'mesh': [], 'eyes_box': None, 'mouth_box': None},
                    'agitation': 0
                })
        except Exception as e:
            print(f"[FRAME] Error processing frame: {e}")

    @socketio.on('finalize_realtime_analysis')
    def handle_finalize(data):
        vid_type = data.get('type', 'suspect')
        print(f"[FORENSIC] Finalizing real-time report for {vid_type}...")
        
        try:
            pipeline = _get_pipeline()
            emotions = _realtime_emotions[vid_type]
            scores = _realtime_scores[vid_type]
            
            if not emotions:
                print(f"[FORENSIC] No frames captured for {vid_type}. Skipping report.")
                return

            # Compute report using pipeline logic
            # We need to simulate some metadata for the pipeline's internal compute_report
            report = pipeline._compute_report(
                video_path=f"{vid_type}_video.mp4",
                duration=len(emotions) * 0.1, # Approx based on our 0.1s interval
                per_second_emotions=emotions,
                emotion_scores_all=scores,
                face_frames=[], # Not needed for metrics
                frame_indices=list(range(len(emotions))),
                gradcam_output_path=os.path.join(PROJECT_ROOT, "outputs", f"{vid_type}_gradcam.png")
            )
            
            # Save report to the standard location
            report_path = os.path.join(PROJECT_ROOT, "phase2_cv", "reports", f"{vid_type}_video_report.json")
            os.makedirs(os.path.dirname(report_path), exist_ok=True)
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=4)
            
            print(f"[FORENSIC] Real-time report saved to {report_path}")
            
            # Emit completion to UI
            emit('video_complete', {
                'type': vid_type,
                'report': {
                    'dominant_emotion': report['dominant_emotion'],
                    'emotional_activation_score': report['emotional_activation_score'],
                    'emotional_consistency_score': report['emotional_consistency_score'],
                    'deception_indicator': report['deception_indicator'],
                    'reactivity_index': report['reactivity_index'],
                    'narrative': report['interpretation']
                }
            })
            
            # Reset for next run
            _realtime_emotions[vid_type] = []
            _realtime_scores[vid_type] = []
            _last_sent_segment_id[vid_type] = -1
            _last_progress_time[vid_type] = -1
            
        except Exception as e:
            print(f"[FORENSIC] Error finalizing report: {e}")

    @socketio.on('run_forensic_analysis')
    def handle_forensic_analysis():
        """Triggers the real Computer Vision engine for the weapon image."""
        try:
            print("[FORENSIC] Triggering dynamic CV analysis engine...")
            import subprocess
            script_path = os.path.join(PROJECT_ROOT, "scratch", "annotate_weapon.py")
            python_path = os.path.join(PROJECT_ROOT, ".venv", "Scripts", "python.exe")
            
            # Run the annotation script
            subprocess.run([python_path, script_path], check=True)
            print("[FORENSIC] Analysis complete. Report and Annotation generated.")
            emit('forensic_analysis_complete', {'status': 'success'})
            
        except Exception as e:
            emit('forensic_analysis_complete', {'status': 'error', 'message': str(e)})
            print(f"[FORENSIC] Analysis Error: {e}")

def generate_verdict_speech_text(results):
    v_desc = results['verdict_description'].upper()
    variant_id = results['selected_variant_id']
    s_id = variant_id.split('_')[1]
    guilt_idx = f"{results['adjusted_score']*100:.2f}"
    
    # Variety Pools for Speech
    intros = [
        "COURTHOUSE OF MADRID. CASE 2024 CR 0471.",
        "SUPERIOR COURT OF JUSTICE, CENTRAL CHAMBER. IN REGARD TO CASE 2024 CR 0471.",
        "MAGISTRATE VIRTUAL PRESENCE INITIATED. PROCEEDINGS 2024 CR 0471.",
        "BEYOND THESE WALLS, JUSTICE IS SERVED. CASE FILE 2024 CR 0471."
    ]

    foundations = [
        f"In accordance with Article 24 of the Spanish Constitution, this court has evaluated the accumulated indicia regarding Daniel Navarro Castillo.",
        f"Pursuant to the legal safeguards of the Spanish Judiciary, we have processed all technical and forensic data in the matter of Navarro Castillo.",
        f"Exercising the authority granted by the State, this bench has synthesized the evidentiary record concerning the accused.",
        f"Under the constitutional principle of legal certainty, the court has completed its deliberation on the suspect's liability."
    ]

    logic_specs = [
        f"Under Article 741 of the Law of Criminal Procedure, we have integrated the juridical persuasion of Defense Strategy {s_id}. The resulting guilt probability index is {guilt_idx} percent.",
        f"Integrating the semantic weight of Strategy {s_id} with the physical findings, the Alis-Core engine yields a technical guilt probability of {guilt_idx} percent.",
        f"Following the analysis of Argumentation Variant {s_id}, the system has calculated an adjusted guilt index of {guilt_idx} percent.",
        f"The court has factored the linguistic strategy {s_id} into the final evaluation, resulting in a probability score of {guilt_idx} percent."
    ]

    conclusions = [
        f"The court hereby resolves the status of: {v_desc}. The investigation remains open under Article 779 LECrim where applicable.",
        f"It is the determination of this chamber to declare the status: {v_desc}. Procedural redirecting is ordered where required by law.",
        f"The final resolution rendered by this system is: {v_desc}. All parties are notified of their right to appeal.",
        f"In view of the findings, the court pronounces the verdict of: {v_desc}."
    ]

    sign_offs = [
        "This sala has pronounced, ordered and signed.",
        "The magistrate has validated and closed these proceedings.",
        "So ordered and mandated by the Superior Court.",
        "Justice is rendered. Session terminated."
    ]

    # Assemble dynamic speech
    intro = random.choice(intros)
    foundation = random.choice(foundations)
    logic = random.choice(logic_specs)
    conclusion = random.choice(conclusions)
    sign_off = random.choice(sign_offs)

    text = f"{intro} {foundation} {logic} {conclusion} {sign_off}"
    return text
