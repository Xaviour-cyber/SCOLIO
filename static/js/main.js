const UI = {
    valState: document.getElementById('val-state'),
    valAngle: document.getElementById('val-angle'),
    valElapsed: document.getElementById('val-elapsed'),
    valFps: document.getElementById('val-fps'),
    gaugeFill: document.getElementById('gauge-fill'),
    timerProgress: document.getElementById('timer-progress'),
    calibrationOverlay: document.getElementById('calibration-overlay'),
    calibrationTimer: document.getElementById('calibration-timer'),
    btnCalibrate: document.getElementById('btn-calibrate'),
    btnSync: document.getElementById('btn-sync'),
    syncIcon: document.getElementById('sync-icon'),
    syncLabel: document.getElementById('sync-label'),
    btnShutdown: document.getElementById('btn-shutdown')
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

    // Update Sync state
    if (data.sync_mode) {
        UI.syncIcon.innerText = "check_circle";
        UI.syncLabel.innerText = "Mode: Sync (Presisi)";
        UI.btnSync.classList.replace('text-primary', 'text-emerald-600');
    } else {
        UI.syncIcon.innerText = "speed";
        UI.syncLabel.innerText = "Mode: Async (Cepat)";
        UI.btnSync.classList.replace('text-emerald-600', 'text-primary');
    }

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

if (UI.btnSync) {
    UI.btnSync.addEventListener('click', async () => {
        try {
            await fetch('/api/toggle_sync', { method: 'POST' });
        } catch (e) {
            console.error("Failed to toggle sync", e);
        }
    });
}

if (UI.btnShutdown) {
    UI.btnShutdown.addEventListener('click', async () => {
        // Change button state
        UI.btnShutdown.innerHTML = '<div class="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin"></div> Menyimpan Grafik...';
        UI.btnShutdown.disabled = true;
        
        try {
            await fetch('/api/shutdown', { method: 'POST' });
        } catch (e) {
            // Error is expected because the server literally kills itself
        }
        
        // Show success screen
        document.body.innerHTML = `
            <div class="min-h-screen flex items-center justify-center bg-slate-900 text-white font-sans text-center px-4">
                <div>
                    <span class="material-symbols-outlined text-6xl text-emerald-500 mb-4 block">check_circle</span>
                    <h1 class="text-3xl font-bold mb-2">Sesi Selesai</h1>
                    <p class="text-slate-400">Data telah disimpan dan grafik scatter plot berhasil dibuat.<br>Server telah dimatikan, Anda dapat menutup jendela ini.</p>
                </div>
            </div>
        `;
    });
}

// Start polling API every 400ms to save CPU
setInterval(fetchStatus, 400);
