document.addEventListener('DOMContentLoaded', () => {
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

    let currentFile = null;

    uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadZone.classList.add('dragover');
    });

    uploadZone.addEventListener('dragleave', () => {
        uploadZone.classList.remove('dragover');
    });

    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadZone.classList.remove('dragover');
        if (e.dataTransfer.files.length) {
            handleFileSelect(e.dataTransfer.files[0]);
        }
    });

    audioInput.addEventListener('change', (e) => {
        if (e.target.files.length) {
            handleFileSelect(e.target.files[0]);
        }
    });

    function handleFileSelect(file) {
        if (!file.name.endsWith('.wav')) {
            alert("Please upload a valid .wav file.");
            return;
        }
        currentFile = file;
        fileName.textContent = file.name;
        uploadZone.classList.add('hidden');
        fileInfo.classList.remove('hidden');
        analyzeBtn.classList.remove('hidden');
    }

    removeFile.addEventListener('click', () => {
        currentFile = null;
        audioInput.value = '';
        uploadZone.classList.remove('hidden');
        fileInfo.classList.add('hidden');
        analyzeBtn.classList.add('hidden');
    });

    analyzeBtn.addEventListener('click', async () => {
        if (!currentFile) return;

        fileInfo.classList.add('hidden');
        analyzeBtn.classList.add('hidden');
        loadingState.classList.remove('hidden');

        try {
            const formData = new FormData();
            formData.append('audio', currentFile);

            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) throw new Error("Server error processing the file.");

            const data = await response.json();
            if(data.error) throw new Error(data.error);

            showResults(data);
        } catch (error) {
            alert("Analysis Failed: " + error.message);
            resetUI();
        }
    });

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
    }

    resetBtn.addEventListener('click', resetUI);

    function resetUI() {
        resultsZone.classList.add('hidden');
        loadingState.classList.add('hidden');
        
        currentFile = null;
        audioInput.value = '';
        uploadZone.classList.remove('hidden');
        fileInfo.classList.add('hidden');
        analyzeBtn.classList.add('hidden');
    }
});
