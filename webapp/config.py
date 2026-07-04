"""
Configuration for the web application
"""

import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent.parent

# Upload settings
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'webapp', 'uploads')
MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500MB max file size
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}

# Model checkpoint paths
CHECKPOINTS_DIR = os.path.join(BASE_DIR, 'checkpoints')

# Few-shot VLM classifier checkpoint (trained by evaluation/train_few_shot_vlm.py,
# linear probe on frozen CLIP embeddings)
FEW_SHOT_VLM_CHECKPOINT = os.path.join(BASE_DIR, 'evaluation', 'checkpoints', 'few_shot_vlm_classifier.pt')

# Supervised fall/no-fall CNN & ViT checkpoints (trained directly on Fall vs. No-Fall
# labels by evaluation/train_cnn_models.py -- distinct from the KTH 6-activity models
# above, which were trained on walking/jogging/running/boxing/handwaving/handclapping).
# Label convention: 0 = No Fall, 1 = Fall (see evaluation/train_cnn_models.py).
EVALUATION_CHECKPOINTS_DIR = os.path.join(BASE_DIR, 'evaluation', 'checkpoints')
FALL_DETECTION_CNN_CHECKPOINTS = {
    '2dcnn_resnet_fall': os.path.join(EVALUATION_CHECKPOINTS_DIR, '2dcnn_resnet_fall_detection.pt'),
    '3dcnn_simple_fall': os.path.join(EVALUATION_CHECKPOINTS_DIR, '3dcnn_simple_fall_detection.pt'),
    'vit_fall': os.path.join(EVALUATION_CHECKPOINTS_DIR, 'vit_fall_detection.pt'),
}
FALL_DETECTION_CLASS_NAMES = ['No Fall', 'Fall']

# Class names (from base_config.yaml)
CLASS_NAMES = [
    'walking',
    'jogging',
    'running',
    'boxing',
    'handwaving',
    'handclapping'
]

# Model checkpoint mapping
# Note: Anomaly detection models are saved with "_anomaly" suffix
# The system will automatically try anomaly checkpoint first if anomaly detection is enabled
MODEL_CHECKPOINTS = {
    '3dcnn_simple': os.path.join(CHECKPOINTS_DIR, '3dcnn_simple_yolov11_best.pth'),
    '3dcnn_deep': os.path.join(CHECKPOINTS_DIR, '3dcnn_deep_yolov11_best.pth'),
    '2dcnn_resnet': os.path.join(CHECKPOINTS_DIR, '2dcnn_resnet_yolov11_best.pth'),
    '2dcnn_lenet': os.path.join(CHECKPOINTS_DIR, '2dcnn_lenet_yolov11_best.pth'),
    '2dcnn': os.path.join(CHECKPOINTS_DIR, '2dcnn_yolov11_best.pth'),  # Backward compatibility
    'vit': os.path.join(CHECKPOINTS_DIR, 'vit_yolov11_best.pth'),
    'stgcn': os.path.join(CHECKPOINTS_DIR, 'stgcn_yolov11_best.pth'),
    'tcnt': os.path.join(CHECKPOINTS_DIR, 'tcnt_yolov11_best.pth'),
}

# Anomaly detection checkpoint mapping (with _anomaly suffix)
ANOMALY_MODEL_CHECKPOINTS = {
    '3dcnn_simple': os.path.join(CHECKPOINTS_DIR, '3dcnn_simple_yolov11_anomaly_best.pth'),
    '3dcnn_deep': os.path.join(CHECKPOINTS_DIR, '3dcnn_deep_yolov11_anomaly_best.pth'),
    '2dcnn_resnet': os.path.join(CHECKPOINTS_DIR, '2dcnn_resnet_yolov11_anomaly_best.pth'),
    '2dcnn_lenet': os.path.join(CHECKPOINTS_DIR, '2dcnn_lenet_yolov11_anomaly_best.pth'),
    '2dcnn': os.path.join(CHECKPOINTS_DIR, '2dcnn_yolov11_anomaly_best.pth'),
    'vit': os.path.join(CHECKPOINTS_DIR, 'vit_yolov11_anomaly_best.pth'),
    'stgcn': os.path.join(CHECKPOINTS_DIR, 'stgcn_yolov11_anomaly_best.pth'),
    'tcnt': os.path.join(CHECKPOINTS_DIR, 'tcnt_yolov11_anomaly_best.pth'),
}


def get_checkpoint_path(model_type: str, use_anomaly_detection: bool = False) -> str:
    """
    Get checkpoint path for a model, preferring anomaly checkpoint if anomaly detection is enabled.
    
    Args:
        model_type: Model type identifier
        use_anomaly_detection: Whether anomaly detection mode is enabled
        
    Returns:
        Path to checkpoint file
    """
    if use_anomaly_detection:
        # Try anomaly checkpoint first
        anomaly_path = ANOMALY_MODEL_CHECKPOINTS.get(model_type)
        if anomaly_path and os.path.exists(anomaly_path):
            return anomaly_path
        # Fallback to regular checkpoint if anomaly doesn't exist
        print(f"Warning: Anomaly checkpoint not found for {model_type}, using regular checkpoint")
    
    # Return regular checkpoint
    return MODEL_CHECKPOINTS.get(model_type, '')

# Preprocessing parameters
CLIP_LENGTH = 32
OVERLAP = 0.5

# Normalization settings (from extraction_config.yaml)
# CRITICAL: Training data was NOT normalized - it uses raw pixel coordinates!
# Set these to False to match training data format
NORMALIZE_SCALE_TO_01 = False  # Changed: Training data uses raw pixel coordinates
NORMALIZE_CENTER_ON_HIP = False  # Changed: Training data uses raw pixel coordinates
HIP_JOINT_INDEX = 8  # COCO format: 8 is mid-hip
IMAGE_WIDTH = 160
IMAGE_HEIGHT = 120

# YOLOv11 settings
YOLOV11_MODEL_NAME = "yolo11n-pose.pt"
YOLOV11_CONFIDENCE = 0.25

# Device
DEVICE = "cuda"  # Will be auto-detected if CUDA available

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

