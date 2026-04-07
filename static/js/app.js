document.addEventListener('DOMContentLoaded', () => {
    // Elements
    const uploadZone = document.getElementById('uploadZone');
    const audioInput = document.getElementById('audioInput');
    const fileInfo = document.getElementById('fileInfo');
    const fileName = document.getElementById('fileName');
    const removeFile = document.getElementById('removeFile');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const loadingState = document.getElementById('loadingState');
    const resultsZone = document.getElementById('resultsZone');
    const resultCard = document.getElementById('resultCard');
    const resetBtn = document.getElementById('resetBtn');
    
    // Feature Elements
    const recordBtn = document.getElementById('recordBtn');
    const stopRecordBtn = document.getElementById('stopRecordBtn');
    const recordingStatus = document.getElementById('recordingStatus');
    const recordTimer = document.getElementById('recordTimer');
    const playPauseWave = document.getElementById('playPauseWave');
    const exportPdfBtn = document.getElementById('exportPdfBtn');
    const uploadPanel = document.getElementById('uploadPanel');
    
    // Tabs
    const navAnalysisBtn = document.getElementById('navAnalysisBtn');
    const navHistoryBtn = document.getElementById('navHistoryBtn');
    const viewAnalysis = document.getElementById('viewAnalysis');
    const viewHistory = document.getElementById('viewHistory');

    navAnalysisBtn.addEventListener('click', () => {
        viewAnalysis.style.display = 'block';
        viewHistory.style.display = 'none';
        navAnalysisBtn.className = 'btn-primary';
        navAnalysisBtn.style.background = '';
        navAnalysisBtn.style.color = 'white';
        navHistoryBtn.className = 'btn-secondary';
        navHistoryBtn.style.background = 'rgba(255,255,255,0.05)';
        navHistoryBtn.style.color = '#cbd5e1';
    });

    navHistoryBtn.addEventListener('click', () => {
        viewAnalysis.style.display = 'none';
        viewHistory.style.display = 'block';
        navHistoryBtn.className = 'btn-primary';
        navHistoryBtn.style.background = '';
        navHistoryBtn.style.color = 'white';
        navAnalysisBtn.className = 'btn-secondary';
        navAnalysisBtn.style.background = 'rgba(255,255,255,0.05)';
        navAnalysisBtn.style.color = '#cbd5e1';
        loadHistory();
    });

    let currentFile = null;
    let wavesurfer = null;
    let radarChartObj = null;

    // --- MediaRecorder setup (PCM encoding) ---
    let audioContext;
    let mediaStream;
    let recorderNode;
    let audioChunks = [];
    let isRecording = false;
    let recTimerInt = null;
    
    // WaveSurfer initialization
    function initWavesurfer(blob) {
        if(wavesurfer) { wavesurfer.destroy(); }
        const urlObj = URL.createObjectURL(blob);
        wavesurfer = WaveSurfer.create({
            container: '#waveform',
            waveColor: 'rgba(0, 210, 255, 0.5)',
            progressColor: '#00d2ff',
            cursorColor: '#10b981',
            barWidth: 2,
            barRadius: 2,
            height: 60,
            url: urlObj
        });
        
        wavesurfer.on('error', (err) => {
            console.log("Wavesurfer native decode exception suppressed.");
        });
        
        playPauseWave.onclick = () => {
             wavesurfer.playPause();
             if(wavesurfer.isPlaying()) {
                  playPauseWave.innerHTML = '<i class="fa-solid fa-pause"></i> Pause Visualizer';
             } else {
                  playPauseWave.innerHTML = '<i class="fa-solid fa-play"></i> Play Audio Visualizer';
             }
        };
        
        wavesurfer.on('finish', () => {
             playPauseWave.innerHTML = '<i class="fa-solid fa-play"></i> Play Audio Visualizer';
        });
    }

    // --- WebAudio API WAV Encoder Utility ---
    function floatTo16BitPCM(output, offset, input){
        for (let i = 0; i < input.length; i++, offset+=2){
            let s = Math.max(-1, Math.min(1, input[i]));
            output.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
        }
    }

    function writeString(view, offset, string){
        for (let i = 0; i < string.length; i++){
            view.setUint8(offset + i, string.charCodeAt(i));
        }
    }

    function encodeWAV(samples, sampleRate){
        const buffer = new ArrayBuffer(44 + samples.length * 2);
        const view = new DataView(buffer);
        // RIFF chunk descriptor
        writeString(view, 0, 'RIFF');
        view.setUint32(4, 36 + samples.length * 2, true);
        writeString(view, 8, 'WAVE');
        // FMT sub-chunk
        writeString(view, 12, 'fmt ');
        view.setUint32(16, 16, true);
        view.setUint16(20, 1, true); // PCM format
        view.setUint16(22, 1, true); // 1 channel (mono)
        view.setUint32(24, sampleRate, true);
        view.setUint32(28, sampleRate * 2, true);
        view.setUint16(32, 2, true);
        view.setUint16(34, 16, true); // 16 bit
        // Data sub-chunk
        writeString(view, 36, 'data');
        view.setUint32(40, samples.length * 2, true);
        // Write PCM samples
        floatTo16BitPCM(view, 44, samples);
        return new Blob([view], { type: 'audio/wav' });
    }
    
    // --- Live Record Control ---
    recordBtn.addEventListener('click', async () => {
        try {
            mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
            audioContext = new (window.AudioContext || window.webkitAudioContext)();
            const source = audioContext.createMediaStreamSource(mediaStream);
            recorderNode = audioContext.createScriptProcessor(4096, 1, 1);
            
            audioChunks = [];
            recorderNode.onaudioprocess = (e) => {
                if(!isRecording) return;
                const channelData = e.inputBuffer.getChannelData(0);
                audioChunks.push(new Float32Array(channelData));
            };
            
            source.connect(recorderNode);
            recorderNode.connect(audioContext.destination);
            
            isRecording = true;
            recordingStatus.classList.remove('hidden');
            let secs = 0;
            recTimerInt = setInterval(() => {
                secs++;
                recordTimer.textContent = `0:0${secs < 10 ? secs : 9}`;
                if(secs >= 6) { stopRecordBtn.click(); } 
            }, 1000);
            
        } catch(e) { alert("Microphone access denied or unavailable."); }
    });
    
    stopRecordBtn.addEventListener('click', () => {
        isRecording = false;
        clearInterval(recTimerInt);
        recorderNode.disconnect();
        mediaStream.getTracks().forEach(t => t.stop());
        recordingStatus.classList.add('hidden');
        recordTimer.textContent = '0:00';
        
        // Flatten
        let length = audioChunks.reduce((acc, val) => acc + val.length, 0);
        let result = new Float32Array(length);
        let offset = 0;
        for (let i = 0; i < audioChunks.length; i++) {
            result.set(audioChunks[i], offset);
            offset += audioChunks[i].length;
        }
        
        const wavBlob = encodeWAV(result, audioContext.sampleRate);
        const file = new File([wavBlob], "live_recording.wav", { type: 'audio/wav' });
        handleFileSelect(file, true);
    });

    // --- File Input ---
    uploadZone.addEventListener('dragover', (e) => { e.preventDefault(); uploadZone.classList.add('dragover'); });
    uploadZone.addEventListener('dragleave', () => { uploadZone.classList.remove('dragover'); });
    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault(); uploadZone.classList.remove('dragover');
        if (e.dataTransfer.files.length) { handleFileSelect(e.dataTransfer.files[0]); }
    });
    audioInput.addEventListener('change', (e) => {
        if (e.target.files.length) handleFileSelect(e.target.files[0]);
    });

    function handleFileSelect(file, isLive = false) {
        if (!file.name.toLowerCase().endsWith('.wav')) { alert("Please supply a .wav file."); return; }
        currentFile = file;
        fileName.textContent = file.name;
        uploadZone.classList.add('hidden');
        fileInfo.classList.remove('hidden');
        analyzeBtn.classList.remove('hidden');
        
        const waveContainer = document.getElementById('waveform-container');
        
        if (isLive) {
            waveContainer.classList.remove('hidden');
            // Force synchronous browser layout reflow so #waveform has a non-zero pixel width
            void fileInfo.offsetWidth;
            
            // Wait 100ms for safety just in case of slow DOM paint
            setTimeout(() => {
                initWavesurfer(file);
            }, 100);
        } else {
            waveContainer.classList.add('hidden');
            if(wavesurfer) { wavesurfer.destroy(); wavesurfer = null; }
        }
    }

    removeFile.addEventListener('click', () => {
        currentFile = null; audioInput.value = '';
        uploadZone.classList.remove('hidden');
        fileInfo.classList.add('hidden');
        analyzeBtn.classList.add('hidden');
        if(wavesurfer) { wavesurfer.destroy(); wavesurfer = null; }
    });

    // --- Fetch Analysis ---
    analyzeBtn.addEventListener('click', async () => {
        if (!currentFile) return;

        const patContact = document.getElementById('patContact');
        if (patContact && patContact.value.trim() !== '') {
            if (patContact.value.replace(/[^0-9]/g, '').length !== 10) {
                alert('Authentication Failed: Please enter exactly a 10-digit contact number.');
                return;
            }
        }

        analyzeBtn.classList.add('hidden');
        removeFile.classList.add('hidden');
        loadingState.classList.remove('hidden');

        try {
            const formData = new FormData();
            formData.append('audio', currentFile);
            formData.append('patName', document.getElementById('patName').value);
            formData.append('patContact', document.getElementById('patContact') ? document.getElementById('patContact').value : '');
            formData.append('patAge', document.getElementById('patAge').value);
            formData.append('patGender', document.getElementById('patGender').value);
            formData.append('patSymptoms', document.getElementById('patSymptoms').value);

            const response = await fetch('/predict', { method: 'POST', body: formData });
            if (!response.ok) throw new Error("Server error.");
            const data = await response.json();
            if(data.error) throw new Error(data.error);
            showResults(data);
            if (typeof loadHistory === 'function') loadHistory();
        } catch (error) {
            alert("Analysis Failed: " + error.message);
            resetUI();
        }
    });

    // --- Build Radar Chart ---
    function renderRadarChart(feats) {
        if (radarChartObj) radarChartObj.destroy();
        const ctx = document.getElementById('radarChart').getContext('2d');
        
        // Normalize constraints for visuals scaling
        const dataValues = [
            Math.min(100, (Math.abs(feats.spectral_centroid) / 150) * 100), 
            Math.min(100, (Math.abs(feats.entropy) / 10) * 100),            
            Math.min(100, (Math.abs(feats.temporal_stability) / 5) * 100),  
            Math.min(100, (Math.abs(feats.cqt_contrast) / 10) * 100)        
        ];

        radarChartObj = new Chart(ctx, {
            type: 'radar',
            data: {
                labels: ['Pitch', 'Messiness', 'Rhythm', 'Sharpness'],
                datasets: [{
                    label: 'Patient Audio Profile',
                    data: dataValues,
                    backgroundColor: 'rgba(0, 210, 255, 0.2)',
                    borderColor: '#00d2ff',
                    pointBackgroundColor: '#5e6ad2',
                    pointBorderColor: '#fff',
                    pointHoverBackgroundColor: '#fff',
                    pointHoverBorderColor: '#10b981',
                    borderWidth: 2,
                }]
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                scales: {
                    r: {
                        angleLines: { color: 'rgba(255, 255, 255, 0.1)' },
                        grid: { color: 'rgba(255, 255, 255, 0.1)' },
                        pointLabels: { color: '#94a3b8', font: {family: 'Inter', size: 14, weight: '600'} },
                        ticks: { display: false, max: 100, min: 0 }
                    }
                },
                plugins: { legend: { display: false } },
                animation: { duration: 1500, easing: 'easeOutQuart' }
            }
        });
    }

    function showResults(data) {
        loadingState.classList.add('hidden');
        resultsZone.classList.remove('hidden');

        resultCard.className = 'glass-panel main-result ' + (data.prediction.toLowerCase());
        document.getElementById('statusIcon').innerHTML = data.prediction === 'Normal'
            ? '<i class="fa-solid fa-heart-circle-check"></i>'
            : '<i class="fa-solid fa-heart-crack"></i>';

        document.getElementById('predictionText').textContent = data.prediction + " Heart Sound";
        document.getElementById('confidenceScore').textContent = (data.confidence * 100).toFixed(2);

        document.getElementById('valSpectral').textContent = data.features.spectral_centroid.toFixed(0) + " Hz";
        document.getElementById('valEntropy').textContent = data.features.entropy.toFixed(2);
        document.getElementById('valTemporal').textContent = data.features.temporal_stability.toFixed(3);
        document.getElementById('valContrast').textContent = data.features.cqt_contrast.toFixed(3);

        renderRadarChart(data.features);

        // ── Populate the printer-friendly report ─────────────
        const isNormal   = data.prediction === 'Normal';
        
        // --- On-Screen Patient Advisory ---
        const advicePanel = document.getElementById('patientAdvicePanel');
        const adviceText = document.getElementById('adviceText');
        const adviceTitle = document.getElementById('adviceTitle');
        advicePanel.style.display = 'block';

        if (isNormal) {
            advicePanel.style.borderLeftColor = '#10b981';
            adviceTitle.style.color = '#10b981';
            adviceText.innerHTML = "<strong>No immediate action required.</strong> Maintain a healthy diet, stay hydrated, and continue regular exercise. Your heart sounds normal. Routine follow-up is recommended.";
        } else {
            advicePanel.style.borderLeftColor = '#ef4444';
            adviceTitle.style.color = '#ef4444';
            adviceText.innerHTML = "<strong>Possible Symptoms:</strong> You may be experiencing shortness of breath, unexplained fatigue, chest pain, or a rapid/irregular heartbeat.<br><br><strong>Recommended Action:</strong> The AI detected acoustic anomalies that may indicate a heart murmur or valvular dysfunction. Please consult a cardiologist to schedule an Echocardiogram.";
        }
        
        const resultColor = isNormal ? '#1b5e20' : '#b71c1c';
        const resultBg    = isNormal ? '#e8f5e9'  : '#ffebee';
        const confidence  = (data.confidence * 100).toFixed(2) + '%';

        document.getElementById('printFileName').textContent =
            currentFile ? currentFile.name : 'Live Recording';
        document.getElementById('printDate').textContent =
            new Date().toLocaleString('en-IN', { dateStyle: 'long', timeStyle: 'short' });

        // Result box dynamic colors
        const box = document.getElementById('printResultBox');
        box.style.background = resultBg;
        box.style.border = `2px solid ${resultColor}`;

        const predEl = document.getElementById('printPrediction');
        predEl.textContent  = data.prediction + ' Heart Sound';
        predEl.style.color  = resultColor;

        const confEl = document.getElementById('printConfidence');
        confEl.textContent  = confidence;
        confEl.style.color  = resultColor;

        document.getElementById('printPitch').textContent     = data.features.spectral_centroid.toFixed(0) + ' Hz';
        document.getElementById('printEntropy').textContent   = data.features.entropy.toFixed(2);
        document.getElementById('printRhythm').textContent    = data.features.temporal_stability.toFixed(3);
        document.getElementById('printSharpness').textContent = data.features.cqt_contrast.toFixed(3);

        document.getElementById('printClinicalNote').textContent = isNormal
            ? 'The analyzed heart sound exhibits regular S1/S2 patterns with no significant murmurs or arrhythmic activity detected. The spectral and temporal features fall within normal reference ranges. Routine follow-up is recommended.'
            : 'The analyzed heart sound shows irregular spectral patterns that may indicate the presence of a murmur, valvular disorder, or arrhythmia. The AI model detected anomalies across multiple acoustic feature domains. Immediate consultation with a cardiologist is strongly advised.';
    }

    resetBtn.addEventListener('click', resetUI);

    function resetUI() {
        resultsZone.classList.add('hidden');
        loadingState.classList.add('hidden');
        uploadPanel.classList.remove('hidden');
        
        currentFile = null;
        audioInput.value = '';
        uploadZone.classList.remove('hidden');
        fileInfo.classList.add('hidden');
        analyzeBtn.classList.add('hidden');
        removeFile.classList.remove('hidden');
        if(wavesurfer) { wavesurfer.destroy(); wavesurfer = null; }
    }

    // ─────────────────────────────────────────────────────────
    // PDF EXPORT  — replace the old exportPdfBtn listener
    // ─────────────────────────────────────────────────────────
    // ─────────────────────────────────────────────────────────
    // PDF EXPORT  — replace the old exportPdfBtn listener
    // ─────────────────────────────────────────────────────────

    
exportPdfBtn.addEventListener('click', async () => {
    // Populate patient information into the report
    document.getElementById('rName').textContent = document.getElementById('patName').value || 'N/A';
    document.getElementById('rAge').textContent = document.getElementById('patAge').value || 'N/A';
    document.getElementById('rGender').textContent = document.getElementById('patGender').value || 'N/A';
    document.getElementById('rSymptoms').textContent = document.getElementById('patSymptoms').value || 'N/A';

    const element = document.getElementById('printableReport');

    // Step 1: Show element properly (VISIBLE in layout)
    // Using left: -9999px instead of opacity: 0 because html2canvas captures opacity!
    element.style.display = 'block';
    element.style.position = 'absolute';
    element.style.left = '-9999px';
    element.style.top = '0';
    element.style.opacity = '1';
    element.style.zIndex = '9999';

    // Step 2: Wait for render + fonts
    await new Promise(resolve => setTimeout(resolve, 500));
    await document.fonts.ready;

    // Step 3: Generate with html2canvas and jsPDF directly
    html2canvas(element, { scale: 2, useCORS: true }).then(canvas => {
        const imgData = canvas.toDataURL("image/png");
        
        const { jsPDF } = window.jspdf;
        const pdf = new jsPDF('p', 'mm', 'a4');
        
        const imgWidth = 210; // A4 width
        const pageHeight = 297; // A4 height
        const imgHeight = canvas.height * imgWidth / canvas.width;
        
        pdf.addImage(imgData, 'PNG', 0, 0, imgWidth, imgHeight);
        pdf.save('CardioAI_Medical_Report.pdf');
        
        // Step 4: Reset
        element.style.display = 'none';
        element.style.position = '';
        element.style.left = '';
        element.style.top = '';
        element.style.opacity = '';
        element.style.zIndex = '';
    });
});

    // --- Patient Database History Fetching ---
    async function loadHistory() {
        try {
            const res = await fetch('/history');
            const data = await res.json();
            const tbody = document.getElementById('historyTableBody');
            if(!tbody) return;
            tbody.innerHTML = '';
            
            data.forEach(record => {
                const tr = document.createElement('tr');
                tr.style.borderBottom = '1px solid rgba(255,255,255,0.05)';
                tr.style.transition = 'background-color 0.2s';
                tr.onmouseover = () => tr.style.backgroundColor = 'rgba(255,255,255,0.05)';
                tr.onmouseout = () => tr.style.backgroundColor = 'transparent';
                
                const riskColor = record.risk_level === 'High' ? '#ef4444' : (record.risk_level === 'Moderate' ? '#f59e0b' : '#10b981');
                const predColor = record.prediction === 'Abnormal' ? '#ef4444' : '#10b981';
                
                tr.innerHTML = `
                    <td style="padding: 12px; font-size: 0.85rem; color: #94a3b8;">${record.date}</td>
                    <td style="padding: 12px; font-weight: 600;">${record.name}</td>
                    <td style="padding: 12px; font-size: 0.85rem; color: #94a3b8;"><i class="fa-solid fa-phone" style="font-size:0.75rem;"></i> ${record.contact || 'N/A'}</td>
                    <td style="padding: 12px; font-size: 0.9rem; color: #cbd5e1;">${record.age} / ${record.gender}</td>
                    <td style="padding: 12px; font-size: 0.85rem; color: #94a3b8;">${record.filename}</td>
                    <td style="padding: 12px; font-weight: bold; color: ${predColor};">${record.prediction}</td>
                    <td style="padding: 12px;">${record.confidence.toFixed(1)}%</td>
                    <td style="padding: 12px;"><span style="background: ${riskColor}33; color: ${riskColor}; padding: 4px 8px; border-radius: 12px; font-size: 0.8rem; font-weight: bold; display: inline-block;">${record.risk_level}</span></td>
                `;
                
                const deleteTd = document.createElement('td');
                deleteTd.style.padding = '12px';
                deleteTd.style.textAlign = 'center';
                
                const delBtn = document.createElement('button');
                delBtn.innerHTML = '<i class="fa-solid fa-trash-can"></i>';
                delBtn.style.cssText = 'background: none; border: none; color: #ef4444; font-size: 1.1rem; cursor: pointer; transition: transform 0.2s;';
                delBtn.title = 'Delete Patient Record';
                delBtn.onmouseover = () => delBtn.style.transform = 'scale(1.2)';
                delBtn.onmouseout = () => delBtn.style.transform = 'scale(1)';
                delBtn.onclick = async () => {
                    if(!confirm("Are you sure you want to permanently delete this patient record?")) return;
                    try {
                        const dr = await fetch(`/history/${record.id}`, { method: 'DELETE' });
                        if(dr.ok) loadHistory();
                    } catch(e) { alert("Failed to delete record: " + e.message); }
                };
                deleteTd.appendChild(delBtn);
                tr.appendChild(deleteTd);
                
                tbody.appendChild(tr);
            });
        } catch(e) {
            console.error("Could not load database history:", e);
        }
    }
    
    // Auto-load history on initialization
    loadHistory();
    
    const refreshHistoryBtn = document.getElementById('refreshHistoryBtn');
    if(refreshHistoryBtn) {
        refreshHistoryBtn.addEventListener('click', loadHistory);
    }

});