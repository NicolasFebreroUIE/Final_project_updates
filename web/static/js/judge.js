const socket = io();

document.addEventListener("DOMContentLoaded", () => {
    const mode = localStorage.getItem("judgeMode") || "single";
    const audioPlayer = document.getElementById("judgeAudio");
    
    // Map targets to DOM elements
    const targets = {
        'primary': { log: document.getElementById('primaryLog'), verdict: document.getElementById('primaryVerdict') },
        'secondary': { log: document.getElementById('worstLog'), verdict: document.getElementById('worstVerdict') } // For compare mode we mapped optimal to primary and worst to secondary
    };
    
    if (mode === "single") {
        document.getElementById("singleModeContainer").classList.remove("hidden");
        const variantId = localStorage.getItem("judgeVariant") || "variant_01";
        
        fetch('/api/get-verdict', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ variant_id: variantId })
        });
        
    } else {
        document.getElementById("compareModeContainer").classList.remove("hidden");
        const opt = localStorage.getItem("judgeVariantOptimal");
        const wst = localStorage.getItem("judgeVariantWorst");
        
        document.getElementById('optimalLabel').innerText = opt;
        document.getElementById('worstLabel').innerText = wst;
        // Remap primary to optimal
        targets['primary'].log = document.getElementById('optimalLog');
        targets['primary'].verdict = document.getElementById('optimalVerdict');
        
        // Backend handles comparing both by feeding 'primary' and 'secondary' workers
        fetch('/api/compare-verdicts', { method: 'POST' });
    }
    
    let currentLines = { 'primary': null, 'secondary': null };
    
    socket.on('verdict_chunk', (data) => {
        const t = targets[data.worker];
        if(!t || !t.log) return;
        
        if (data.text === '\n') {
            currentLines[data.worker] = null;
        } else {
            if (!currentLines[data.worker]) {
                currentLines[data.worker] = document.createElement('div');
                currentLines[data.worker].className = 'verdict-line';
                t.log.appendChild(currentLines[data.worker]);
            }
            currentLines[data.worker].innerHTML += data.text;
            t.log.parentElement.scrollTop = t.log.parentElement.scrollHeight;
        }
    });
    
    const getVerdictClass = (verdict) => {
        if(verdict === 'GUILTY') return 'verdict-guilty';
        if(verdict === 'PROBABLE_GUILT') return 'verdict-inconclusive'; // Map probable guilt to warning colors
        if(verdict === 'INCONCLUSIVE') return 'verdict-inconclusive';
        return 'verdict-insufficient';
    };
    
    socket.on('verdict_complete', (data) => {
        const t = targets[data.worker];
        if(!t) return;
        
        t.verdict.innerText = data.verdict.replace(/_/g, ' ');
        t.verdict.className = `verdict-text ${getVerdictClass(data.verdict)}`;
        
        if(mode === "single") {
            document.getElementById("paradoxNote").classList.remove("hidden");
        }
    });
    
    socket.on('audio_ready', (data) => {
        audioPlayer.src = data.path;
        audioPlayer.play().catch(e => console.log("Audio autoplay prevented by browser. User interaction needed.", e));
    });
});
