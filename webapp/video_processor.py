"""
Video processing pipeline for web application
Handles skeleton extraction and preprocessing
"""

import os
import sys
import numpy as np
import torch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from skeleton_extraction import extract_with_yolov11
from preprocessing import generate_clips, create_2d_image_representation

# Import webapp config
sys.path.insert(0, os.path.dirname(__file__))
from config import (
    CLIP_LENGTH, OVERLAP, NORMALIZE_SCALE_TO_01, NORMALIZE_CENTER_ON_HIP,
    HIP_JOINT_INDEX, IMAGE_WIDTH, IMAGE_HEIGHT, YOLOV11_MODEL_NAME, YOLOV11_CONFIDENCE
)


def normalize_keypoints(keypoints: np.ndarray) -> np.ndarray:
    """
    Normalize keypoints according to extraction config.
    MUST MATCH the normalization used during training!
    
    CRITICAL: Training data was saved with RAW pixel coordinates (not normalized).
    Therefore, normalization is DISABLED (NORMALIZE_SCALE_TO_01=False, NORMALIZE_CENTER_ON_HIP=False)
    to match the training data format.
    
    Args:
        keypoints: Array of shape (T, J, C) with raw coordinates
        
    Returns:
        Keypoints array (normalized if config enabled, otherwise raw)
    """
    T, J, C = keypoints.shape
    keypoints = keypoints.copy()
    
    # CRITICAL: Order matters! Center FIRST, then scale (matches training)
    if NORMALIZE_CENTER_ON_HIP:
        # Use average of left_hip (11) and right_hip (12) as reference
        # This matches the training normalization in src/skeleton_extraction.py
        if J >= 13:  # Ensure we have enough joints
            hip_coords = (keypoints[:, 11, :] + keypoints[:, 12, :]) / 2.0  # Average of both hips
            keypoints = keypoints - hip_coords[:, np.newaxis, :]  # (T, J, C)
        else:
            # Fallback to single hip joint if not enough joints
            hip_coords = keypoints[:, HIP_JOINT_INDEX:HIP_JOINT_INDEX+1, :].copy()  # (T, 1, C)
            keypoints = keypoints - hip_coords
    
    if NORMALIZE_SCALE_TO_01:
        # Scale coordinates to [0, 1] by image dimensions (AFTER centering)
        keypoints[:, :, 0] = keypoints[:, :, 0] / IMAGE_WIDTH  # x coordinates
        keypoints[:, :, 1] = keypoints[:, :, 1] / IMAGE_HEIGHT  # y coordinates
    
    return keypoints


def extract_and_preprocess_video(video_path: str, model_type: str) -> list:
    """
    Extract skeleton from video and preprocess into clips ready for model inference.
    
    Args:
        video_path: Path to video file
        model_type: Model type ('3dcnn_simple', '3dcnn_deep', '2dcnn', 'vit', 'stgcn', 'tcnt')
        
    Returns:
        List of preprocessed clips, each ready for model input
    """
    # Extract skeleton using YOLOv11 (now returns tuple: keypoints, bounding_boxes)
    print(f"Extracting skeleton from {video_path}...")
    extraction_result = extract_with_yolov11(
        video_path=video_path,
        model_name=YOLOV11_MODEL_NAME,
        confidence=YOLOV11_CONFIDENCE
    )
    
    # Handle both old (array) and new (tuple) return formats
    if isinstance(extraction_result, tuple):
        keypoints, bounding_boxes = extraction_result
    else:
        keypoints = extraction_result
        bounding_boxes = None
    
    if keypoints is None or (hasattr(keypoints, 'shape') and keypoints.shape[0] == 0):
        raise ValueError("Failed to extract skeleton from video or no person detected")
    
    print(f"Extracted {keypoints.shape[0]} frames with {keypoints.shape[1]} joints")
    
    # Normalize keypoints
    keypoints = normalize_keypoints(keypoints)
    
    # Generate clips
    clips = generate_clips(keypoints, clip_length=CLIP_LENGTH, overlap=OVERLAP)
    print(f"Generated {len(clips)} clips")
    
    # Preprocess each clip based on model type
    preprocessed_clips = []
    for clip in clips:
        # Convert to tensor and reshape based on model type
        if model_type in ['3dcnn_simple', '3dcnn_deep']:
            # 3D CNN: [C, J, T]
            skeleton_tensor = torch.from_numpy(clip).float()  # (T, J, C)
            skeleton_tensor = skeleton_tensor.permute(2, 1, 0)  # (C, J, T)
            
        elif model_type in ['2dcnn_resnet', '2dcnn_lenet', '2dcnn']:
            # 2D CNN: [C=1, H=T, W=2J]
            image_2d = create_2d_image_representation(clip)  # (T, 2J)
            skeleton_tensor = torch.from_numpy(image_2d).float()  # (T, 2J)
            skeleton_tensor = skeleton_tensor.unsqueeze(0)  # (1, T, 2J)
            
        elif model_type == 'vit':
            # ViT: [C=1, H=T, W=2J] (2D image like 2D CNN)
            image_2d = create_2d_image_representation(clip)  # (T, 2J)
            skeleton_tensor = torch.from_numpy(image_2d).float()  # (T, 2J)
            skeleton_tensor = skeleton_tensor.unsqueeze(0)  # (1, T, 2J)
        elif model_type in ['stgcn', 'tcnt']:
            # ST-GCN, TCNTE: [C, J, T]
            skeleton_tensor = torch.from_numpy(clip).float()  # (T, J, C)
            skeleton_tensor = skeleton_tensor.permute(2, 1, 0)  # (C, J, T)
            
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        
        preprocessed_clips.append(skeleton_tensor)
    
    return preprocessed_clips

