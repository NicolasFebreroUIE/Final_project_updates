// Use the global 'socket' instance initialized in base.html
document.addEventListener("DOMContentLoaded", () => {
    console.log("[VIDEOS] DOM Content Loaded. Forensics UI active.");
    
    if (typeof socket === 'undefined') {
        console.warn("[VIDEOS] Global socket not found. Attempting local init...");
        try {
            window.socket = io();
        } catch(e) { console.error("[VIDEOS] Socket init failed:", e); }
    }
    
    let completedCount = 0;

    // 1. ATTACH BUTTON LOGIC FIRST (CRITICAL)
    const setupButton = (btnId, vidId) => {
        const btn = document.getElementById(btnId);
        const vid = document.getElementById(vidId);
        if (btn && vid) {
            btn.onclick = () => {
                console.log(`[VIDEOS] Button ${btnId} CLICKED. Starting playback for ${vidId}`);
                btn.style.backgroundColor = "#ff4a4a";
                btn.innerText = "[ ANALYSIS INITIATED ]";
                
                vid.play().then(() => {
                    console.log(`[VIDEOS] Playback started for ${vidId}`);
                }).catch(e => {
                    console.error("[VIDEOS] Playback failed:", e);
                    btn.innerText = "[ PLAYBACK ERROR ]";
                    btn.style.backgroundColor = "var(--danger)";
                    alert("Playback failed. Please ensure the video files are correctly served and your browser supports standard MP4.");
                });
            };
        } else {
            console.error(`[VIDEOS] Could not find elements: ${btnId} or ${vidId}`);
        }
    };
    setupButton('startSuspectAnalysis', 'suspectVideo');
    setupButton('startWitnessAnalysis', 'witnessVideo');

    // 2. INITIALIZE CHARTS WITH SAFETY
    let sChart = null;
    let wChart = null;

    try {
        if (typeof Chart !== 'undefined') {
            const chartConfig = {
                type: 'line',
                data: { 
                    labels: [], 
                    datasets: [
                        { label: 'Angry', data: [], borderColor: '#ff4444', borderWidth: 1, pointRadius: 0, fill: false },
                        { label: 'Happy', data: [], borderColor: '#00ff88', borderWidth: 1, pointRadius: 0, fill: false },
                        { label: 'Sad', data: [], borderColor: '#4a9eff', borderWidth: 1, pointRadius: 0, fill: false },
                        { label: 'Neutral', data: [], borderColor: '#888888', borderWidth: 1, pointRadius: 0, fill: false }
                    ] 
                },
                options: { 
                    responsive: true, 
                    maintainAspectRatio: false, 
                    scales: { 
                        x: { display: false },
                        y: { min: 0, max: 1, grid: { color: '#222' } } 
                    }, 
                    plugins: { legend: { labels: { color: '#888', font: { size: 10 } } } },
                    animation: false 
                }
            };
            const sCtx = document.getElementById('sChart').getContext('2d');
            const wCtx = document.getElementById('wChart').getContext('2d');
            sChart = new Chart(sCtx, JSON.parse(JSON.stringify(chartConfig)));
            wChart = new Chart(wCtx, JSON.parse(JSON.stringify(chartConfig)));
            console.log("[VIDEOS] Charts initialized.");
        } else {
            console.warn("[VIDEOS] Chart.js not loaded. Visual graphs will be skipped.");
        }
    } catch (err) {
        console.error("[VIDEOS] Chart initialization error:", err);
    }
    
    const updateProgress = (prefix, act, con, dec) => {
        const elAct = document.getElementById(`${prefix}ActVal`);
        const elActB = document.getElementById(`${prefix}ActBar`);
        if (elAct) elAct.innerText = Boolean(act) ? act.toFixed(2) : "0.00";
        if (elActB) elActB.style.width = `${(act || 0) * 100}%`;
        
        const elCon = document.getElementById(`${prefix}ConVal`);
        const elConB = document.getElementById(`${prefix}ConBar`);
        if (elCon) elCon.innerText = Boolean(con) ? con.toFixed(2) : "0.00";
        if (elConB) elConB.style.width = `${(con || 0) * 100}%`;
        
        const elDec = document.getElementById(`${prefix}DecVal`);
        const elDecB = document.getElementById(`${prefix}DecBar`);
        if (elDec) elDec.innerText = Boolean(dec) ? dec.toFixed(2) : "0.00";
        if (elDecB) {
            elDecB.style.width = `${(dec || 0) * 100}%`;
            elDecB.style.backgroundColor = dec > 0.6 ? 'var(--danger)' : 'var(--success)';
        }
    };

    function attachVideoLogic(type) {
        const video = document.getElementById(`${type}Video`);
        const overlay = document.getElementById(`${type}Overlay`);
        if (!video || !overlay) return;

        const prefix = type === 'suspect' ? 's' : 'w';
        const emotionLabel = document.getElementById(`${type}EmotionLabel`);
        
        let ctx = overlay.getContext('2d');
        let offscreenCanvas = document.createElement('canvas');
        let offCtx = offscreenCanvas.getContext('2d');
        offscreenCanvas.width = 640;
        offscreenCanvas.height = 360;

        let isAnalyzing = false;
        let lastAnalysisTime = -1;
        let analysisInterval = 0.1; // Reduced for GPU/Optimized responsiveness
        
        video.addEventListener('play', () => {
            if (!isAnalyzing) {
                console.log(`[VIDEOS] Notifying server of analysis start: ${type}`);
                fetch('/api/analyze-video', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({type: type, real_time: true})
                }).catch(e => console.error("[VIDEOS] API call failed:", e));
                isAnalyzing = true;
                if (emotionLabel) emotionLabel.innerText = '[ANALYSIS ACTIVE]';
            }
            overlay.width = video.clientWidth;
            overlay.height = video.clientHeight;
            requestAnimationFrame(processLoop);
        });

        function processLoop() {
            if (video.paused || video.ended) return;
            
            let currentTime = video.currentTime;
            if (currentTime - lastAnalysisTime >= analysisInterval) {
                lastAnalysisTime = currentTime;
                
                const scale = 640 / video.videoWidth;
                offscreenCanvas.width = 640;
                offscreenCanvas.height = video.videoHeight * scale;
                
                offCtx.drawImage(video, 0, 0, offscreenCanvas.width, offscreenCanvas.height);
                let b64 = offscreenCanvas.toDataURL('image/jpeg', 0.5);
                socket.emit('process_single_frame', { 
                    type: type, 
                    image: b64, 
                    time: currentTime,
                    width: offscreenCanvas.width,
                    height: offscreenCanvas.height
                });
            }
            requestAnimationFrame(processLoop);
        }
        
        video.addEventListener('ended', () => {
            console.log(`[VIDEOS] Video ended: ${type}. Finalizing analysis.`);
            socket.emit('finalize_realtime_analysis', { type: type });
        });
    }

    attachVideoLogic('suspect');
    attachVideoLogic('witness');

    // Silent Pre-transcription trigger
    if (typeof socket !== 'undefined') {
        socket.emit('init_transcription', { type: 'suspect' });
        socket.emit('init_transcription', { type: 'witness' });
    }

    // Persistence Check
    ['suspect', 'witness'].forEach(type => {
        if (localStorage.getItem(`video_${type}_synced`) === 'true') {
            const btn = document.getElementById(`sync${type.charAt(0).toUpperCase() + type.slice(1)}`);
            if (btn) {
                btn.classList.remove('hidden');
                btn.innerText = "[ DATA SYNCHRONIZED ]";
                btn.style.borderColor = "var(--success)";
                btn.disabled = true;
                completedCount++;
            }
        }
    });
    if (completedCount >= 2) {
        document.getElementById('nextBtn').classList.remove('hidden');
    }

    socket.on('frame_emotion', (data) => {
        const overlay = document.getElementById(`${data.type}Overlay`);
        const video = document.getElementById(`${data.type}Video`);
        if (!overlay || !video) return;
        
        const ctx = overlay.getContext('2d');
        const prefix = data.type === 'suspect' ? 's' : 'w';
        
        // Update UI Labels
        const emotionEl = document.getElementById(`${data.type}EmotionLabel`);
        if (emotionEl) emotionEl.innerText = data.emotion.toUpperCase();
        
        // Update Activity Gauges
        if (data.behavioral) {
            const agVal = Math.min(1.0, data.agitation * 2.0);
            updateProgress(prefix, data.confidence, 1.0 - (data.behavioral.ear || 0.3), agVal);
        }

        // Update Charts
        const chart = data.type === 'suspect' ? sChart : wChart;
        if (chart) {
            const labels = ['angry', 'happy', 'sad', 'neutral'];
            labels.forEach((label, idx) => {
                const labelIdx = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'].indexOf(label);
                const prob = data.probs[labelIdx] || 0;
                chart.data.datasets[idx].data.push(prob);
                if (chart.data.datasets[idx].data.length > 50) chart.data.datasets[idx].data.shift();
            });
            if (chart.data.labels.length < 50) chart.data.labels.push("");
            chart.update('none');
        }

        // Clear overlay
        ctx.clearRect(0, 0, overlay.width, overlay.height);
        const w = overlay.width;
        const h = overlay.height;

        // --- FORENSIC MESH: Dots + High-Tech Lines ---
        if (data.behavioral && data.behavioral.mesh && data.behavioral.mesh.length > 0) {
            const mesh = data.behavioral.mesh;
            const AMBER = '#FFB000';
            const CYAN = '#00f2ff';
            const color = data.type === 'suspect' ? CYAN : AMBER;

            ctx.fillStyle = color;
            ctx.strokeStyle = color;
            ctx.lineWidth = 0.5;

            // 1. Draw points
            for (let i = 0; i < mesh.length; i += 15) { // Sparse points
                const x = mesh[i] * w;
                const y = mesh[i+1] * h;
                ctx.beginPath();
                ctx.arc(x, y, 1, 0, 2 * Math.PI);
                ctx.fill();
            }

            // 2. Draw connections (Simplified high-tech triangulation)
            ctx.globalAlpha = 0.2;
            for (let i = 0; i < mesh.length - 30; i += 30) {
                const x1 = mesh[i] * w;
                const y1 = mesh[i+1] * h;
                const x2 = mesh[i+15] * w;
                const y2 = mesh[i+16] * h;
                ctx.beginPath();
                ctx.moveTo(x1, y1);
                ctx.lineTo(x2, y2);
                ctx.stroke();
            }
            
            // 3. Scanning line effect
            const scanY = (Date.now() % 2000) / 2000 * h;
            ctx.globalAlpha = 0.4;
            ctx.beginPath();
            ctx.moveTo(0, scanY);
            ctx.lineTo(w, scanY);
            ctx.lineWidth = 1;
            ctx.stroke();
            
            ctx.globalAlpha = 1.0;
        }

        // --- FORENSIC BOXES: Eyes & Mouth (AMBER #FFB000) ---
        const AMBER = '#FFB000';
        ctx.strokeStyle = AMBER;
        ctx.lineWidth = 2;

        // Scale factors from 640x360 analysis buffer to current overlay display size
        const rx = w / 640;
        const ry = h / (360 * (h/w) / (360/640) || 360); // Adjust ry based on aspect ratio safety
        
        // Refined ry: since we always scale to 640 width, the height in analysis is 640 * (vidH/vidW)
        const analysisH = 640 * (video.videoHeight / video.videoWidth);
        const RY = h / analysisH;
        const RX = w / 640;

        if (data.behavioral && data.behavioral.eyes_box) {
            data.behavioral.eyes_box.forEach(eyeBox => {
                if (!eyeBox) return;
                const [x1, y1, x2, y2] = eyeBox;
                ctx.strokeRect(x1 * RX, y1 * RY, (x2 - x1) * RX, (y2 - y1) * RY);
                // Corner accents
                const cLen = 6;
                ctx.beginPath();
                ctx.moveTo(x1*RX, y1*RY + cLen); ctx.lineTo(x1*RX, y1*RY); ctx.lineTo(x1*RX + cLen, y1*RY);
                ctx.moveTo(x2*RX - cLen, y1*RY); ctx.lineTo(x2*RX, y1*RY); ctx.lineTo(x2*RX, y1*RY + cLen);
                ctx.stroke();
            });
        }

        if (data.behavioral && data.behavioral.mouth_box) {
            const [mx1, my1, mx2, my2] = data.behavioral.mouth_box;
            ctx.strokeRect(mx1 * RX, my1 * RY, (mx2 - mx1) * RX, (my2 - my1) * RY);
            // Label
            ctx.fillStyle = AMBER;
            ctx.font = '10px monospace';
            ctx.fillText('ORAL', mx1 * RX, my1 * RY - 3);
        }

        // --- FACE BOUNDING BOX ---
        if (data.bbox) {
            const [bx, by, bw, bh] = data.bbox;
            ctx.strokeStyle = data.emotion === 'angry' ? '#FF4444' : '#00FF88';
            ctx.lineWidth = 1;
            ctx.strokeRect(bx*RX, by*RY, bw*RX, bh*RY);
            
            // Emotion label on face box
            ctx.fillStyle = ctx.strokeStyle;
            ctx.font = 'bold 11px monospace';
            ctx.fillText(`${data.emotion.toUpperCase()} ${(data.confidence*100).toFixed(0)}%`, bx*RX, by*RY - 5);
        }
    });

    socket.on('transcription_chunk', (data) => {
        console.log(`[TRANSCRIPT] Chunk received for ${data.type}:`, data.text);
        const box = document.getElementById(`${data.type}Transcript`);
        if(box.innerHTML.includes("[Awaiting transcription...]")) {
            // Keep the header, clear the placeholder
            const header = box.querySelector('.forensic-log-header');
            const headerHTML = header ? header.outerHTML : '';
            box.innerHTML = headerHTML;
        }
        if(!data.is_final && data.text) {
            box.innerHTML += `<span>${data.text}</span> `;
            box.scrollTop = box.scrollHeight;
        }
    });

    socket.on('video_complete', (data) => {
        const prefix = data.type === 'suspect' ? 's' : 'w';
        const report = data.report;
        
        document.getElementById(`${data.type}EmotionLabel`).innerText = report.dominant_emotion.toUpperCase();
        updateProgress(prefix, report.emotional_activation_score, report.emotional_consistency_score, report.deception_indicator);
        
        const conc = document.getElementById(`${data.type}Conclusion`);
        conc.innerHTML = `
            <div style="border-bottom: 1px solid var(--border-color); padding-bottom: 5px; margin-bottom: 10px;">
                <strong style="color: var(--success);">FORENSIC BEHAVIORAL REPORT</strong>
            </div>
            <p style="font-size: 0.95rem; line-height: 1.4; color: #eee; margin-bottom: 15px;">
                ${report.narrative}
            </p>
            <div style="font-size: 0.85rem; color: #888; display: grid; grid-template-columns: 1fr 1fr; gap: 5px;">
                <span>Dominant: ${report.dominant_emotion.toUpperCase()}</span>
                <span>Activation: ${report.emotional_activation_score.toFixed(2)}</span>
                <span>Consistency: ${report.emotional_consistency_score.toFixed(2)}</span>
                <span>Deception: <b style="color: ${report.deception_indicator > 0.6 ? 'var(--danger)' : 'var(--success)'}">${report.deception_indicator > 0.6 ? 'HIGH' : 'LOW'}</b> (${report.deception_indicator.toFixed(2)})</span>
            </div>
        `;
        conc.classList.remove('hidden');
        
        const syncBtn = document.getElementById(`sync${data.type.charAt(0).toUpperCase() + data.type.slice(1)}`);
        if (syncBtn) {
            syncBtn.classList.remove('hidden');
            syncBtn.onclick = () => {
                syncBtn.innerText = "[ DATA SYNCHRONIZED ]";
                syncBtn.style.borderColor = "var(--success)";
                syncBtn.disabled = true;
                
                localStorage.setItem(`video_${data.type}_synced`, 'true');
                completedCount++;
                if(completedCount >= 2) {
                    document.getElementById('nextBtn').classList.remove('hidden');
                }
            };
        }
    });
});
