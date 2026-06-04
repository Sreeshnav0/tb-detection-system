document.addEventListener('DOMContentLoaded', () => {
    const uploadArea = document.getElementById('upload-area');
    const fileInput = document.getElementById('file-input');
    const imagePreview = document.getElementById('image-preview');
    const placeholder = document.getElementById('placeholder');
    const uploadBtn = document.getElementById('upload-btn');
    const loading = document.getElementById('loading');
    const resetBtn = document.getElementById('reset-btn');

    // UI Elements for Results
    const predictionText = document.getElementById('prediction-text');
    const probabilityText = document.getElementById('probability-text');
    const tbProbabilityText = document.getElementById('tb-probability-text');
    const riskText = document.getElementById('risk-text');
    const gradcamImage = document.getElementById('gradcam-image');
    const originalCrop = document.getElementById('original-crop');
    const fileNameDisplay = document.getElementById('file-name-display');
    const analysisReportSection = document.getElementById('analysis-report-section');

    // Layout Toggles
    const visualPlaceholder = document.getElementById('visual-placeholder');
    const resultsView = document.getElementById('results-view');

    // Handle Upload Area Click
    uploadArea.addEventListener('click', () => fileInput.click());

    // Handle File Selection
    fileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            fileNameDisplay.textContent = file.name;
            const reader = new FileReader();
            reader.onload = (event) => {
                imagePreview.src = event.target.result;
                imagePreview.style.display = 'block';
                placeholder.style.display = 'none';
                uploadBtn.disabled = false;

                // Set initial state for analyze button
                uploadBtn.style.backgroundColor = '#2b459d';
                analysisReportSection.style.display = 'none';
            };
            reader.readAsDataURL(file);
        }
    });

    // Handle Analyze Click
    uploadBtn.addEventListener('click', async () => {
        const file = fileInput.files[0];
        if (!file) return;

        const formData = new FormData();
        formData.append('file', file);

        // UI State: Loading
        uploadBtn.disabled = true;
        visualPlaceholder.style.display = 'none';
        resultsView.style.display = 'none';
        loading.style.display = 'block';

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) throw new Error('Prediction failed');

            const data = await response.json();
            displayResults(data);

        } catch (error) {
            console.error('Error:', error);
            alert('Error analyzing the X-ray image. Please try again.');
            visualPlaceholder.style.display = 'block';
        } finally {
            loading.style.display = 'none';
            uploadBtn.disabled = false;
        }
    });

    function displayResults(data) {
        predictionText.textContent = data.prediction;
        probabilityText.textContent = `${(parseFloat(data.confidence) * 100).toFixed(2)}%`;

        if (tbProbabilityText) {
            tbProbabilityText.textContent = `${(parseFloat(data.tb_probability) * 100).toFixed(2)}%`;
        }

        const riskIndicator = document.getElementById('risk-indicator');
        riskText.textContent = data.risk_level;

        // Apply color coding
        riskIndicator.className = 'risk-indicator-box'; // reset
        if (data.risk_level.includes('Low')) {
            riskIndicator.classList.add('low-risk');
        } else if (data.risk_level.includes('High')) {
            riskIndicator.classList.add('high-risk');
        } else if (data.risk_level.includes('Moderate')) {
            riskIndicator.classList.add('moderate-risk');
        }
        riskIndicator.style.display = 'block';

        // Visual Explanations (Right Side)
        const timestamp = new Date().getTime();
        gradcamImage.src = `${data.gradcam_image}?t=${timestamp}`;
        originalCrop.src = imagePreview.src;

        // UI Refinement: Hide left preview after analysis
        imagePreview.style.display = 'none';
        placeholder.style.display = 'block';

        resultsView.style.display = 'block';
        analysisReportSection.style.display = 'block';
    }

    // Reset Analysis (If you want a reset button, otherwise it's just triggered by new upload)
    if (resetBtn) {
        resetBtn.addEventListener('click', () => {
            fileInput.value = '';
            imagePreview.style.display = 'none';
            placeholder.style.display = 'block';
            fileNameDisplay.textContent = '';
            uploadBtn.disabled = true;
            uploadBtn.style.backgroundColor = '#94a3b8';

            resultsView.style.display = 'none';
            visualPlaceholder.style.display = 'block';
            analysisReportSection.style.display = 'none';
        });
    }

    // Drag and Drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.style.borderColor = '#2563eb';
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.style.borderColor = '#e2e8f0';
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        const file = e.dataTransfer.files[0];
        if (file && file.type.startsWith('image/')) {
            fileInput.files = e.dataTransfer.files;
            const event = new Event('change');
            fileInput.dispatchEvent(event);
        }
    });
});
