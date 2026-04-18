const socket = io();

document.addEventListener("DOMContentLoaded", () => {
    const log = document.getElementById("trainingLog");
    const nextBtn = document.getElementById("nextBtn");
    
    // Setup charts
    const lossCtx = document.getElementById('lossChart').getContext('2d');
    const accCtx = document.getElementById('accChart').getContext('2d');
    
    const lossChart = new Chart(lossCtx, {
        type: 'line',
        data: { labels: [], datasets: [{ label: 'Loss', data: [], borderColor: '#f39c12', tension: 0.1 }] },
        options: { responsive: true, maintainAspectRatio: false, scales: { y: { min: 0 } } }
    });
    
    const accChart = new Chart(accCtx, {
        type: 'line',
        data: { labels: [], datasets: [{ label: 'Accuracy %', data: [], borderColor: '#4a9eff', tension: 0.1 }] },
        options: { responsive: true, maintainAspectRatio: false, scales: { y: { min: 0, max: 100 } } }
    });
    
    function logMsg(msg) {
        log.innerHTML += msg + "<br>";
        log.scrollTop = log.scrollHeight;
    }
    
    socket.on('training_log', (data) => {
        logMsg(data.text);
    });
    
    // Manual trigger button
    document.getElementById('startActionBtn').onclick = (e) => {
        e.target.style.display = 'none';
        logMsg("[SYSTEM] Initializing background compute thread...");
        fetch('/api/start-training', { method: 'POST' });
    };
    
    socket.on('training_progress', (data) => {
        let bar = "█".repeat(Math.round((data.epoch/data.total_epochs)*10)) + "░".repeat(10 - Math.round((data.epoch/data.total_epochs)*10));
        
        // Only log to terminal if it's fresh mode, replay mode gets its own specific log events from backend
        if (!data.replay) {
            logMsg(`[TRAINING] Epoch ${data.epoch}/${data.total_epochs} ${bar} ${Math.round((data.epoch/data.total_epochs)*100)}% | Loss: ${data.loss.toFixed(4)} | Acc: ${data.accuracy.toFixed(1)}%`);
        }
        
        lossChart.data.labels.push(data.epoch);
        lossChart.data.datasets[0].data.push(data.loss);
        lossChart.update();
        
        accChart.data.labels.push(data.epoch);
        accChart.data.datasets[0].data.push(data.accuracy);
        accChart.update();
    });
    
    socket.on('training_complete', (data) => {
        if (data.mode === 'replay') {
            document.getElementById('cache-badge').style.display = 'block';
        }
        
        logMsg("");
        logMsg("[MODEL] Training complete.");
        logMsg("[MODEL] Model saved to models/central_model.pkl");
        logMsg("[MODEL] This model will now be used for ALL subsequent evaluations.");
        logMsg("[MODEL] It will not be modified again.");
        
        document.getElementById('summaryBox').classList.remove('hidden');
        document.getElementById('finalLoss').textContent = data.final_loss.toFixed(4);
        document.getElementById('finalAcc').textContent = data.final_accuracy.toFixed(1) + "%";
        
        nextBtn.classList.remove('hidden');
    });
});
