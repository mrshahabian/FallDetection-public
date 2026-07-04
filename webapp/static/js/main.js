// Global state
let currentFilename = null;
let currentModel = null;

// DOM elements
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const uploadStatus = document.getElementById('uploadStatus');
const videoSection = document.getElementById('videoSection');
const videoPlayer = document.getElementById('videoPlayer');
const modelsSection = document.getElementById('modelsSection');
const fallCnnSection = document.getElementById('fallCnnSection');
const vlmSection = document.getElementById('vlmSection');
const resultsSection = document.getElementById('resultsSection');
const loadingOverlay = document.getElementById('loadingOverlay');
const modelButtons = document.querySelectorAll('.model-btn:not(.vlm-btn)');
const vlmButtons = document.querySelectorAll('.vlm-btn');
const probabilitiesContainer = document.getElementById('probabilitiesContainer');
const currentModelSpan = document.getElementById('currentModel');
const predictedClassSpan = document.getElementById('predictedClass');
const predictionConfidenceSpan = document.getElementById('predictionConfidence');

// Upload area click handler
uploadArea.addEventListener('click', () => {
    fileInput.click();
});

// File input change handler
fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFileUpload(e.target.files[0]);
    }
});

// Drag and drop handlers
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    
    if (e.dataTransfer.files.length > 0) {
        handleFileUpload(e.dataTransfer.files[0]);
    }
});

// Handle file upload
function handleFileUpload(file) {
    // Validate file extension (more reliable than MIME types)
    const allowedExtensions = ['mp4', 'avi', 'mov', 'mkv', 'webm'];
    const fileName = file.name.toLowerCase();
    const fileExtension = fileName.split('.').pop();
    
    if (!allowedExtensions.includes(fileExtension)) {
        showUploadStatus(`Invalid file type. Allowed formats: ${allowedExtensions.join(', ').toUpperCase()}`, 'error');
        return;
    }
    
    // Also check MIME type as secondary validation (but don't fail if it's empty)
    if (file.type && file.type !== '') {
        const allowedMimeTypes = ['video/mp4', 'video/x-msvideo', 'video/avi', 'video/quicktime', 'video/x-matroska', 'video/webm'];
        const hasValidMimeType = allowedMimeTypes.some(type => file.type.toLowerCase().includes(type.split('/')[1]));
        // Only warn if MIME type doesn't match, but don't block upload
        if (!hasValidMimeType) {
            console.warn(`File MIME type (${file.type}) doesn't match expected types, but extension is valid. Proceeding with upload.`);
        }
    }

    // Validate file size (500MB max)
    const maxSize = 500 * 1024 * 1024;
    if (file.size > maxSize) {
        showUploadStatus('File too large. Maximum size is 500MB.', 'error');
        return;
    }

    // Show loading
    showLoading('Uploading video...');

    // Create FormData
    const formData = new FormData();
    formData.append('file', file);

    // Upload file
    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        hideLoading();
        if (data.success) {
            // Store original filename for predictions
            currentFilename = data.filename;
            
            // Use playback_filename for video player (converted version if available)
            const playbackFilename = data.playback_filename || data.filename;
            
            // Show appropriate message
            let statusMessage = 'Video uploaded successfully!';
            if (data.was_converted) {
                statusMessage = 'Video uploaded and converted for playback!';
            }
            showUploadStatus(statusMessage, 'success');
            
            // Display video using playback filename
            displayVideo(playbackFilename);
            showModelsSection();
            hideResults();
        } else {
            showUploadStatus(data.error || 'Upload failed', 'error');
        }
    })
    .catch(error => {
        hideLoading();
        showUploadStatus('Upload failed: ' + error.message, 'error');
        console.error('Upload error:', error);
    });
}

// Display video
function displayVideo(filename) {
    videoSection.style.display = 'block';
    
    // Hide any fallback message from previous uploads
    const fallbackMsg = document.getElementById('videoFallback');
    if (fallbackMsg) {
        fallbackMsg.style.display = 'none';
    }
    
    // Set video source
    const videoUrl = `/video/${filename}`;
    videoPlayer.src = videoUrl;
    videoPlayer.style.display = 'block';
    
    // Reset error handler
    videoPlayer.onerror = null;
    videoPlayer.onloadeddata = null;
    
    // Handle video load error
    videoPlayer.onerror = function(e) {
        console.error('Video playback error:', e, videoUrl);
        
        // Show fallback message if video can't play
        videoPlayer.style.display = 'none';
        
        let fallbackEl = document.getElementById('videoFallback');
        if (!fallbackEl) {
            fallbackEl = document.createElement('div');
            fallbackEl.id = 'videoFallback';
            fallbackEl.className = 'video-fallback';
            videoPlayer.parentNode.appendChild(fallbackEl);
        }
        
        fallbackEl.innerHTML = `
            <div class="fallback-content">
                <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <rect x="2" y="2" width="20" height="20" rx="2" ry="2"></rect>
                    <polyline points="10 8 16 12 10 16"></polyline>
                </svg>
                <p class="fallback-title">Video Preview Not Available</p>
                <p class="fallback-text">
                    Your video has been uploaded successfully and can be processed for activity recognition.
                    However, this format could not be played in your browser.
                </p>
                <p class="fallback-note">
                    <strong>Note:</strong> You can still use all prediction models. The video will be processed correctly.
                </p>
            </div>
        `;
        fallbackEl.style.display = 'block';
    };
    
    // Clear error handler on successful load
    videoPlayer.onloadeddata = function() {
        console.log('Video loaded successfully:', filename);
        videoPlayer.style.display = 'block';
        const fallbackEl = document.getElementById('videoFallback');
        if (fallbackEl) {
            fallbackEl.style.display = 'none';
        }
    };
    
    // Also handle canplay event
    videoPlayer.oncanplay = function() {
        console.log('Video can play:', filename);
        videoPlayer.style.display = 'block';
        const fallbackEl = document.getElementById('videoFallback');
        if (fallbackEl) {
            fallbackEl.style.display = 'none';
        }
    };
    
    videoPlayer.load();
}

// Show models section
function showModelsSection() {
    modelsSection.style.display = 'block';
    fallCnnSection.style.display = 'block';
    vlmSection.style.display = 'block';
}

// Hide results
function hideResults() {
    resultsSection.style.display = 'none';
    currentModel = null;
    modelButtons.forEach(btn => btn.classList.remove('active'));
    vlmButtons.forEach(btn => btn.classList.remove('active'));
}

// Model button click handlers (skeleton-based models)
modelButtons.forEach(btn => {
    btn.addEventListener('click', () => {
        const modelType = btn.getAttribute('data-model');
        runPrediction(modelType);
    });
});

// VLM button click handlers
vlmButtons.forEach(btn => {
    btn.addEventListener('click', () => {
        const vlmMode = btn.getAttribute('data-vlm-mode');
        runVLMPrediction(vlmMode);
    });
});

// Run prediction
function runPrediction(modelType) {
    if (!currentFilename) {
        alert('Please upload a video first');
        return;
    }

    // Update UI
    currentModel = modelType;
    modelButtons.forEach(btn => {
        if (btn.getAttribute('data-model') === modelType) {
            btn.classList.add('active');
        } else {
            btn.classList.remove('active');
        }
    });

    showLoading('Processing video with ' + modelType + '...');
    resultsSection.style.display = 'block';

    // Check if anomaly detection is enabled
    const anomalyToggle = document.getElementById('anomalyDetectionToggle');
    const useAnomalyDetection = anomalyToggle ? anomalyToggle.checked : false;
    
    // Send prediction request
    fetch(`/predict/${modelType}`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            filename: currentFilename,
            use_anomaly_detection: useAnomalyDetection
        })
    })
    .then(response => response.json())
    .then(data => {
        hideLoading();
        if (data.success) {
            displayResults(data);
        } else {
            showError(data.error || 'Prediction failed');
        }
    })
    .catch(error => {
        hideLoading();
        showError('Prediction failed: ' + error.message);
        console.error('Prediction error:', error);
    });
}

// Run VLM prediction
function runVLMPrediction(vlmMode) {
    if (!currentFilename) {
        alert('Please upload a video first');
        return;
    }

    // Update UI
    currentModel = `vlm_${vlmMode}`;
    vlmButtons.forEach(btn => {
        if (btn.getAttribute('data-vlm-mode') === vlmMode) {
            btn.classList.add('active');
        } else {
            btn.classList.remove('active');
        }
    });
    modelButtons.forEach(btn => btn.classList.remove('active'));

    const modeLabels = {
        zero_shot: 'Zero-Shot Fall Detection',
        few_shot: 'Few-Shot Fall Detection',
        video_description: 'Window Description'
    };
    const modeName = modeLabels[vlmMode] || vlmMode;
    showLoading(`Processing video with VLM ${modeName}...`);
    resultsSection.style.display = 'block';

    // Send VLM prediction request
    fetch(`/vlm/${vlmMode}`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            filename: currentFilename
        })
    })
    .then(response => response.json())
    .then(data => {
        hideLoading();
        if (data.success) {
            displayVLMResults(data, vlmMode);
        } else {
            showError(data.error || 'VLM prediction failed');
        }
    })
    .catch(error => {
        hideLoading();
        showError('VLM prediction failed: ' + error.message);
        console.error('VLM prediction error:', error);
    });
}

// Display results
function displayResults(data) {
    // Handle simple fall detection results
    if (data.model === 'simple_fall') {
        displaySimpleFallResults(data);
        return;
    }

    const probabilities = data.probabilities;
    const predictedClass = data.predicted_class;
    const confidence = data.confidence;
    const anomalyDetection = data.anomaly_detection;

    // Update header
    const modelDisplayNames = {
        '2dcnn_resnet_fall': '2D CNN ResNet (Fall Detection)',
        '3dcnn_simple_fall': '3D CNN Simple (Fall Detection)',
        'vit_fall': 'Vision Transformer (Fall Detection)'
    };
    currentModelSpan.textContent = (modelDisplayNames[data.model] || data.model).toUpperCase();
    
    // Display anomaly detection result if available
    if (anomalyDetection) {
        const isFall = anomalyDetection.is_fall;
        const anomalyScore = anomalyDetection.anomaly_score;
        const threshold = anomalyDetection.threshold;
        
        predictedClassSpan.textContent = isFall ? '⚠️ FALL DETECTED' : '✓ NO FALL';
        predictedClassSpan.style.color = isFall ? '#dc3545' : '#28a745';
        predictionConfidenceSpan.textContent = `Anomaly Score: ${(anomalyScore * 100).toFixed(1)}% (Threshold: ${(threshold * 100).toFixed(1)}%)`;
    } else {
        predictedClassSpan.textContent = predictedClass;
        predictedClassSpan.style.color = '';
        predictionConfidenceSpan.textContent = `(${(confidence * 100).toFixed(1)}%)`;
    }

    // Display annotated video if available
    if (data.annotated_video) {
        console.log('Displaying annotated video:', data.annotated_video);
        displayVideo(data.annotated_video);
    }

    // Clear previous results
    probabilitiesContainer.innerHTML = '';

    // Display anomaly detection section if available
    if (anomalyDetection) {
        const anomalyItem = document.createElement('div');
        anomalyItem.className = 'probability-item';
        anomalyItem.style.border = anomalyDetection.is_fall ? '3px solid #dc3545' : '3px solid #28a745';
        anomalyItem.style.background = anomalyDetection.is_fall ? '#fff5f5' : '#f0fff4';
        anomalyItem.style.marginBottom = '20px';
        
        const isFall = anomalyDetection.is_fall;
        const anomalyScore = anomalyDetection.anomaly_score;
        const threshold = anomalyDetection.threshold;
        const conf = anomalyDetection.confidence;
        const entropy = anomalyDetection.entropy;
        
        anomalyItem.innerHTML = `
            <div style="text-align: center; padding: 20px;">
                <div style="font-size: 3em; margin-bottom: 10px;">${isFall ? '⚠️' : '✓'}</div>
                <div style="font-size: 1.5em; font-weight: bold; color: ${isFall ? '#dc3545' : '#28a745'}; margin-bottom: 15px;">
                    ${isFall ? 'FALL DETECTED' : 'NO FALL DETECTED'}
                </div>
                <div style="font-size: 1.1em; color: #666; margin-bottom: 20px;">
                    <div style="margin: 8px 0;">
                        <strong>Anomaly Score:</strong> <span style="color: ${isFall ? '#dc3545' : '#28a745'};">${(anomalyScore * 100).toFixed(2)}%</span>
                    </div>
                    <div style="margin: 8px 0;">
                        <strong>Threshold:</strong> ${(threshold * 100).toFixed(2)}%
                    </div>
                    <div style="margin: 8px 0;">
                        <strong>Confidence:</strong> ${(conf * 100).toFixed(2)}%
                    </div>
                    <div style="margin: 8px 0;">
                        <strong>Entropy:</strong> ${entropy.toFixed(4)}
                    </div>
                </div>
                <div class="probability-bar-container" style="max-width: 400px; margin: 0 auto;">
                    <div class="probability-bar" style="width: ${Math.min(anomalyScore * 100, 100)}%; background: ${isFall ? '#dc3545' : '#28a745'};">
                        ${(anomalyScore * 100).toFixed(1)}%
                    </div>
                </div>
            </div>
        `;
        probabilitiesContainer.appendChild(anomalyItem);
    }

    // Sort probabilities by value (descending)
    const sortedProbs = Object.entries(probabilities)
        .sort((a, b) => b[1] - a[1]);

    // Create probability items
    sortedProbs.forEach(([className, prob]) => {
        const item = document.createElement('div');
        item.className = 'probability-item';
        if (className === predictedClass && !anomalyDetection) {
            item.classList.add('highest');
        }

        const percentage = (prob * 100).toFixed(2);

        item.innerHTML = `
            <div class="probability-header">
                <span class="class-name">${className}</span>
                <span class="probability-value">${percentage}%</span>
            </div>
            <div class="probability-bar-container">
                <div class="probability-bar" style="width: ${percentage}%">
                    ${percentage}%
                </div>
            </div>
        `;

        probabilitiesContainer.appendChild(item);
    });

    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Display simple fall detection results
function displaySimpleFallResults(data) {
    const isFall = data.is_fall;
    const fallProb = data.fall_probability || 0;
    const numWindows = data.num_windows || 0;
    const windowsWithFall = data.windows_with_fall || 0;
    
    // Update header
    currentModelSpan.textContent = 'SIMPLE FALL DETECTION';
    predictedClassSpan.textContent = isFall ? '⚠️ FALL DETECTED' : '✓ NO FALL';
    predictedClassSpan.style.color = isFall ? '#ff4444' : '#44ff44';
    predictionConfidenceSpan.textContent = `(${(fallProb * 100).toFixed(1)}%)`;
    
    // Display annotated video if available
    if (data.annotated_video) {
        console.log('Displaying annotated video:', data.annotated_video);
        displayVideo(data.annotated_video);
    }
    
    // Clear previous results
    probabilitiesContainer.innerHTML = '';
    
    // Fall probability item
    const fallItem = document.createElement('div');
    fallItem.className = 'probability-item';
    if (isFall) {
        fallItem.classList.add('highest');
    }
    
    const fallPercentage = (fallProb * 100).toFixed(2);
    const noFallPercentage = ((1 - fallProb) * 100).toFixed(2);
    
    fallItem.innerHTML = `
        <div class="probability-header">
            <span class="class-name">${isFall ? '⚠️ FALL DETECTED' : '✓ NO FALL'}</span>
            <span class="probability-value">${fallPercentage}%</span>
        </div>
        <div class="probability-bar-container">
            <div class="probability-bar" style="width: ${fallPercentage}%; background-color: ${isFall ? '#ff4444' : '#44ff44'}">
                ${fallPercentage}%
            </div>
        </div>
    `;
    probabilitiesContainer.appendChild(fallItem);
    
    // Statistics
    const statsItem = document.createElement('div');
    statsItem.className = 'probability-item';
    statsItem.style.marginTop = '15px';
    statsItem.style.padding = '15px';
    statsItem.style.backgroundColor = '#f5f5f5';
    statsItem.style.borderRadius = '5px';
    statsItem.innerHTML = `
        <div style="font-weight: bold; margin-bottom: 10px;">Statistics:</div>
        <div style="margin: 5px 0;"><strong>Total Windows:</strong> ${numWindows}</div>
        <div style="margin: 5px 0;"><strong>Windows with Fall:</strong> ${windowsWithFall}</div>
        <div style="margin: 5px 0;"><strong>Fall Rate:</strong> ${numWindows > 0 ? ((windowsWithFall / numWindows) * 100).toFixed(1) : 0}%</div>
    `;
    probabilitiesContainer.appendChild(statsItem);
    
    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Display VLM results
function displayVLMResults(data, vlmMode) {
    // Update header
    let modeName = 'VLM';
    if (vlmMode === 'zero_shot') {
        modeName = 'VLM Zero-Shot Fall Detection';
    } else if (vlmMode === 'few_shot') {
        modeName = 'VLM Few-Shot Fall Detection';
    } else if (vlmMode === 'video_description') {
        modeName = 'VLM Window Description';
    }
    currentModelSpan.textContent = modeName.toUpperCase();

    // Clear previous results
    probabilitiesContainer.innerHTML = '';

    if (vlmMode === 'zero_shot' || vlmMode === 'few_shot') {
        // Handle window-based fall detection (zero-shot or few-shot)
        const fallProb = data.fall_probability || (data.result ? data.result.fall_probability : 0);
        const isFall = data.is_fall !== undefined ? data.is_fall : (data.result ? data.result.is_fall : false);
        const numWindows = data.num_windows || 0;
        const windowsWithFall = data.windows_with_fall || 0;
        
        // Display annotated video if available
        if (data.annotated_video) {
            console.log('Displaying annotated video:', data.annotated_video);
            displayVideo(data.annotated_video);
        }
        
        // Update header with clear result
        predictedClassSpan.textContent = isFall ? 'Fall Detected' : 'No Fall';
        predictionConfidenceSpan.textContent = `(${(fallProb * 100).toFixed(1)}%)`;
        
        // Create main result display
        const fallItem = document.createElement('div');
        fallItem.className = `probability-item ${isFall ? 'highest' : ''}`;
        fallItem.style.border = isFall ? '3px solid #dc3545' : '3px solid #28a745';
        fallItem.style.background = isFall ? '#fff5f5' : '#f0fff4';
        
        const statusIcon = isFall ? '⚠️' : '✓';
        const statusText = isFall ? 'FALL DETECTED' : 'NO FALL DETECTED';
        const statusColor = isFall ? '#dc3545' : '#28a745';
        
        fallItem.innerHTML = `
            <div style="text-align: center; padding: 20px;">
                <div style="font-size: 3em; margin-bottom: 10px;">${statusIcon}</div>
                <div style="font-size: 1.5em; font-weight: bold; color: ${statusColor}; margin-bottom: 15px;">
                    ${statusText}
                </div>
                <div style="font-size: 1.2em; color: #666; margin-bottom: 20px;">
                    Fall Probability: <strong style="color: ${statusColor};">${(fallProb * 100).toFixed(2)}%</strong>
                </div>
                <div class="probability-bar-container" style="max-width: 400px; margin: 0 auto;">
                    <div class="probability-bar" style="width: ${fallProb * 100}%; background: ${statusColor};">
                        ${(fallProb * 100).toFixed(1)}%
                    </div>
                </div>
            </div>
        `;
        probabilitiesContainer.appendChild(fallItem);
        
        // Show statistics if available (from window-based detection)
        if (numWindows > 0) {
            const statsItem = document.createElement('div');
            statsItem.className = 'probability-item';
            statsItem.style.marginTop = '15px';
            statsItem.style.padding = '15px';
            statsItem.style.backgroundColor = '#f5f5f5';
            statsItem.style.borderRadius = '5px';
            const methodLine = vlmMode === 'zero_shot' && data.num_prompts
                ? `<div style="margin: 5px 0;"><strong>Prompt Bank:</strong> ${data.num_prompts} balanced fall/non-fall prompts</div>`
                : (vlmMode === 'few_shot'
                    ? `<div style="margin: 5px 0;"><strong>Classifier:</strong> linear probe on frozen CLIP embeddings</div>`
                    : '');
            statsItem.innerHTML = `
                <div style="font-weight: bold; margin-bottom: 10px;">Window Analysis:</div>
                <div style="margin: 5px 0;"><strong>Total Windows:</strong> ${numWindows}</div>
                <div style="margin: 5px 0;"><strong>Windows with Fall:</strong> ${windowsWithFall}</div>
                <div style="margin: 5px 0;"><strong>Fall Rate:</strong> ${((windowsWithFall / numWindows) * 100).toFixed(1)}%</div>
                ${methodLine}
            `;
            probabilitiesContainer.appendChild(statsItem);
        }
        
    } else if (vlmMode === 'video_description') {
        // Display window-based video description
        const description = data.description || 'No description available';
        const numWindows = data.num_windows || 0;
        const perWindow = data.per_window || [];
        
        // Update header
        predictedClassSpan.textContent = 'Video Description';
        predictionConfidenceSpan.textContent = `${numWindows} windows analyzed`;
        
        // Display overall description
        const descItem = document.createElement('div');
        descItem.className = 'probability-item';
        descItem.style.padding = '20px';
        descItem.style.backgroundColor = '#f8f9fa';
        descItem.style.borderRadius = '8px';
        descItem.style.marginBottom = '15px';
        descItem.innerHTML = `
            <div style="font-weight: bold; font-size: 1.1em; margin-bottom: 10px; color: #333;">
                Overall Description:
            </div>
            <div style="font-size: 1em; line-height: 1.6; color: #555; padding: 15px; background: white; border-radius: 5px; border-left: 4px solid #007bff;">
                ${description}
            </div>
        `;
        probabilitiesContainer.appendChild(descItem);
        
        // Display per-window descriptions if available
        if (perWindow.length > 0) {
            const windowsHeader = document.createElement('div');
            windowsHeader.style.fontWeight = 'bold';
            windowsHeader.style.fontSize = '1.1em';
            windowsHeader.style.marginTop = '20px';
            windowsHeader.style.marginBottom = '10px';
            windowsHeader.textContent = 'Per-Window Descriptions:';
            probabilitiesContainer.appendChild(windowsHeader);
            
            perWindow.forEach((window, idx) => {
                const windowItem = document.createElement('div');
                windowItem.className = 'probability-item';
                windowItem.style.padding = '12px';
                windowItem.style.marginBottom = '8px';
                windowItem.style.backgroundColor = '#ffffff';
                windowItem.style.border = '1px solid #dee2e6';
                windowItem.style.borderRadius = '5px';
                windowItem.innerHTML = `
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px;">
                        <span style="font-weight: bold; color: #495057;">Window ${window.window_num}</span>
                        <span style="font-size: 0.9em; color: #6c757d;">Frames ${window.window_start}-${window.window_end}</span>
                    </div>
                    <div style="color: #212529; font-size: 0.95em;">${window.description}</div>
                `;
                probabilitiesContainer.appendChild(windowItem);
            });
        }
        
    }

    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Show error
function showError(message) {
    probabilitiesContainer.innerHTML = `
        <div class="upload-status error" style="display: block;">
            ${message}
        </div>
    `;
}

// Show upload status
function showUploadStatus(message, type) {
    uploadStatus.textContent = message;
    uploadStatus.className = `upload-status ${type}`;
    setTimeout(() => {
        uploadStatus.style.display = 'none';
    }, 5000);
}

// Show loading overlay
function showLoading(text = 'Processing...') {
    const loadingText = document.querySelector('.loading-text');
    if (loadingText) {
        loadingText.textContent = text;
    }
    loadingOverlay.style.display = 'flex';
}

// Hide loading overlay
function hideLoading() {
    loadingOverlay.style.display = 'none';
}

