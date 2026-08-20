const UI = {
    valState: document.getElementById('val-state'),
    valAngle: document.getElementById('val-angle'),
    valElapsed: document.getElementById('val-elapsed'),
    valFps: document.getElementById('val-fps'),
    gaugeFill: document.getElementById('gauge-fill'),
    timerProgress: document.getElementById('timer-progress'),
    calibrationOverlay: document.getElementById('calibration-overlay'),
    calibrationTimer: document.getElementById('calibration-timer'),
    btnCalibrate: document.getElementById('btn-calibrate')
};

// Apple Color Palette used in the Tailwind design
const COLORS = {
    NORMAL: '#34C759',    // Apple Green
    WARNING: '#FF9500',   // Apple Orange
    ALERT: '#FF3B30',     // Apple Red
    CALIBRATING: '#3B82F6',// Apple Blue
    PRIMARY: '#030304'    // Tailwind Primary
};

async function fetchStatus() {
    try {
        const response = await fetch('/api/status');
        const data = await response.json();
        updateUI(data);
    } catch (error) {
        console.error("Error fetching status:", error);
    }
}

function updateUI(data) {
    // Update FPS
    UI.valFps.innerText = data.fps.toFixed(0);

    if (data.state === "CALIBRATING") {
        UI.calibrationOverlay.classList.remove('hidden');
        UI.calibrationTimer.innerText = data.calibration_remaining.toFixed(1);
        
        UI.valState.innerText = "Kalibrasi";
        UI.valState.style.color = COLORS.CALIBRATING;
        
        UI.valAngle.innerText = "0.0°";
        UI.valElapsed.innerText = "0.0s";
        UI.gaugeFill.style.width = "0%";
        UI.timerProgress.style.width = "0%";
        
    } else {
        UI.calibrationOverlay.classList.add('hidden');
        
        // Update Angle
        UI.valAngle.innerText = data.angle.toFixed(1) + "°";
        // Gauge logic: max visually at 15 degrees. 10 deg = 66.6%.
        let gaugePct = Math.min((data.angle / 15.0) * 100, 100);
        UI.gaugeFill.style.width = gaugePct + "%";

        // Update State
        let stateTitle = data.state.charAt(0).toUpperCase() + data.state.slice(1).toLowerCase();
        UI.valState.innerText = stateTitle;
        
        let color = COLORS.NORMAL;
        if (data.state === "WARNING") color = COLORS.WARNING;
        if (data.state === "ALERT") color = COLORS.ALERT;
        
        UI.valState.style.color = color;
        UI.gaugeFill.style.backgroundColor = color;

        // Update Timer
        UI.valElapsed.innerText = data.elapsed.toFixed(1) + "s";
        UI.timerProgress.style.width = (data.timer_progress * 100) + "%";
        
        // Timer color transitions
        if (data.timer_progress < 0.5) {
            UI.timerProgress.style.backgroundColor = COLORS.PRIMARY; // Standard load color
        } else if (data.timer_progress < 0.8) {
            UI.timerProgress.style.backgroundColor = COLORS.WARNING;
        } else {
            UI.timerProgress.style.backgroundColor = COLORS.ALERT;
        }
    }
}

// Event Listeners
if (UI.btnCalibrate) {
    UI.btnCalibrate.addEventListener('click', async () => {
        try {
            await fetch('/api/calibrate', { method: 'POST' });
        } catch (e) {
            console.error("Failed to calibrate", e);
        }
    });
}

// Start polling API every 150ms
setInterval(fetchStatus, 150);
