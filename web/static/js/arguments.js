const socket = io();

document.addEventListener("DOMContentLoaded", () => {
    let selectedVariant = null;
    let optimalVariantId = null;
    let worstVariantId = null;
    
    const log = document.getElementById("scoringLog");
    const vList = document.getElementById("variantList");
    
    // Setup Chart
    const ctx = document.getElementById('scoresChart').getContext('2d');
    const chart = new Chart(ctx, {
        type: 'bar',
        data: { labels: [], datasets: [{ label: 'NLP Score', data: [], backgroundColor: [] }] },
        options: { 
            indexAxis: 'y', 
            responsive: true, 
            maintainAspectRatio: false,
            scales: { x: { min: 0, max: 1 } },
            plugins: {
                tooltip: { callbacks: { title: function(ctx) { return ctx[0].label; } } }
            }
        }
    });
    
    const getVerdictColor = (verdict) => {
        if(verdict === 'GUILTY') return '#e74c3c';
        if(verdict === 'PROBABLE_GUILT') return '#f39c12';
        if(verdict === 'INCONCLUSIVE') return '#f1c40f'; // yellow
        return '#2ecc71';
    };
    
    socket.on('scoring_progress', (data) => {
        let bar = "█".repeat(Math.round(data.score*10)) + "░".repeat(10 - Math.round(data.score*10));
        log.innerHTML += `Scoring ${data.variant_id}... ${data.score.toFixed(4)}  ${bar}<br>`;
        log.scrollTop = log.scrollHeight;
        
        // Update Chart
        chart.data.labels.push(data.variant_id);
        chart.data.datasets[0].data.push(data.score);
        chart.data.datasets[0].backgroundColor.push(getVerdictColor(data.verdict));
        chart.update();
        
        // Add card
        const card = document.createElement('div');
        card.className = "card";
        card.style.cursor = "pointer";
        card.style.transition = "background-color 0.2s";
        card.innerHTML = `
            <div style="display: flex; justify-content: space-between;">
                <strong>${data.variant_id}</strong>
                <span style="color: ${getVerdictColor(data.verdict)}; font-weight: bold;">${data.score.toFixed(4)}</span>
            </div>
            <div style="font-size: 0.8em; color: #888; margin-top: 5px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">
                ${data.text}
            </div>
        `;
        
        card.onclick = () => {
            // Unselect all
            document.querySelectorAll("#variantList .card").forEach(c => {
                c.style.borderColor = "var(--border-color)";
                c.style.backgroundColor = "var(--panel-bg)";
            });
            // Select this
            card.style.borderColor = "var(--accent)";
            card.style.backgroundColor = "#242d38";
            selectedVariant = data.variant_id;
            
            document.getElementById('btnJudgeSelected').innerHTML = `[ SEND ${selectedVariant} TO JUDGE ]`;
            document.getElementById('btnJudgeSelected').style.borderColor = "var(--accent)";
            document.getElementById('btnJudgeSelected').style.color = "var(--accent)";
        };
        
        vList.appendChild(card);
    });

    socket.on('scoring_complete', (data) => {
        document.getElementById('paradoxBox').classList.remove('hidden');
        document.getElementById('actionArea').classList.remove('hidden');
        document.getElementById('crossingsCount').innerText = data.crossings;
        optimalVariantId = data.optimal;
        worstVariantId = data.worst;
    });
    
    // Start backend evaluation on button click
    document.getElementById('startActionBtn').onclick = (e) => {
        e.target.style.display = 'none';
        log.innerHTML = '[INITIALIZING NLP SCORING PIPELINE]...<br>';
        fetch('/api/score-arguments', { method: 'POST' });
    };
    
    // Handlers
    document.getElementById('btnJudgeSelected').onclick = () => {
        if(!selectedVariant) {
            alert("Please select a variant to send to the judge.");
            return;
        }
        localStorage.setItem("judgeMode", "single");
        localStorage.setItem("judgeVariant", selectedVariant);
        window.location.href = "/judge";
    };
    
    document.getElementById('btnJudgeCompare').onclick = () => {
        localStorage.setItem("judgeMode", "compare");
        localStorage.setItem("judgeVariantOptimal", optimalVariantId || "variant_02");
        localStorage.setItem("judgeVariantWorst", worstVariantId || "variant_06");
        window.location.href = "/judge";
    };
});
