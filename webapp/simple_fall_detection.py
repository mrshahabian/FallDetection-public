"""
Simple fall detection based on skeleton pose analysis
Detects falls by checking if the human body is horizontal
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import sys
import os
# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))
from preprocessing import generate_clips

# Import webapp config
sys.path.insert(0, os.path.dirname(__file__))
from config import CLIP_LENGTH, OVERLAP


def detect_fall_in_keypoints(keypoints: np.ndarray, confidence_threshold: float = 0.3) -> bool:
    """
    Detect if a person has fallen based on keypoint positions.
    
    A fall is detected if:
    1. The body is mostly horizontal (head and hips at similar height)
    2. The vertical extent (height) is much smaller than horizontal extent (width)
    3. The body orientation is close to horizontal
    
    Args:
        keypoints: Array of shape (J, 2) or (T, J, 2) with keypoint coordinates
        confidence_threshold: Minimum confidence (not used for simple detection, kept for compatibility)
        
    Returns:
        True if fall is detected, False otherwise
    """
    if keypoints.ndim == 3:
        # (T, J, 2) - use average across frames
        keypoints = np.mean(keypoints, axis=0)  # Average to (J, 2)
    
    # COCO 17 keypoint indices
    NOSE = 0
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16
    
    # Get key body points
    nose = keypoints[NOSE]
    left_shoulder = keypoints[LEFT_SHOULDER]
    right_shoulder = keypoints[RIGHT_SHOULDER]
    left_hip = keypoints[LEFT_HIP]
    right_hip = keypoints[RIGHT_HIP]
    left_ankle = keypoints[LEFT_ANKLE]
    right_ankle = keypoints[RIGHT_ANKLE]
    
    # Check if keypoints are valid (not all zeros)
    if np.all(nose == 0) or np.all(left_hip == 0) or np.all(right_hip == 0):
        return False  # Invalid pose
    
    # Calculate body center points
    shoulder_center = (left_shoulder + right_shoulder) / 2.0
    hip_center = (left_hip + right_hip) / 2.0
    ankle_center = (left_ankle + right_ankle) / 2.0
    
    # Method 1: Check if body is horizontal
    # Calculate vertical distance between head and hips
    head_hip_vertical_dist = float(abs(nose[1] - hip_center[1]))  # Y coordinate difference
    head_hip_horizontal_dist = float(abs(nose[0] - hip_center[0]))  # X coordinate difference
    
    # If horizontal distance is much larger than vertical, body is horizontal
    if head_hip_horizontal_dist > 0:
        vertical_ratio = head_hip_vertical_dist / head_hip_horizontal_dist
        # If vertical ratio is small (< 0.5), body is mostly horizontal
        if vertical_ratio < 0.5:
            return True
    
    # Method 2: Check body extent (width vs height)
    # Get bounding box of body
    all_points = np.array([
        nose, left_shoulder, right_shoulder, 
        left_hip, right_hip, left_ankle, right_ankle
    ])
    
    # Filter out zero points
    valid_points = all_points[np.any(all_points != 0, axis=1)]
    if len(valid_points) < 3:
        return False
    
    min_x, min_y = np.min(valid_points, axis=0)
    max_x, max_y = np.max(valid_points, axis=0)
    
    body_width = max_x - min_x
    body_height = max_y - min_y
    
    if body_height > 0:
        aspect_ratio = float(body_width / body_height)
        # If width is much larger than height (aspect ratio > 2.0), body is horizontal
        if aspect_ratio > 2.0:
            return True
    
    # Method 3: Check if hips are at similar height to head
    # (indicating person is lying down)
    if body_height > 0:
        height_ratio = float(head_hip_vertical_dist / body_height)
        # If head and hips are close in height (< 30% of body height), likely fallen
        if height_ratio < 0.3:
            return True
    
    return False


def predict_simple_fall_detection(
    video_path: str,
    clip_length: int = 32,
    overlap: float = 0.5,
    return_per_window: bool = False
) -> Dict:
    """
    Perform simple fall detection on a video using skeleton pose analysis.
    
    Args:
        video_path: Path to video file
        clip_length: Length of each time window in frames
        overlap: Overlap ratio between windows
        return_per_window: If True, return per-window results
        
    Returns:
        Dictionary with fall detection results:
        - 'is_fall': Boolean indicating if fall detected (averaged)
        - 'fall_probability': Probability of fall (0.0 to 1.0)
        - 'per_window': List of per-window results (if return_per_window=True)
    """
    # Import video processor (add path if needed)
    sys.path.insert(0, os.path.dirname(__file__))
    from video_processor import extract_with_yolov11
    from config import YOLOV11_MODEL_NAME, YOLOV11_CONFIDENCE
    
    # Extract skeleton keypoints
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
        raise ValueError("Failed to extract skeleton from video")
    
    # Generate time windows (clips)
    clips = generate_clips(keypoints, clip_length=clip_length, overlap=overlap)
    
    if len(clips) == 0:
        raise ValueError("No clips generated from video")
    
    # Calculate window step
    step = int(clip_length * (1 - overlap))
    step = max(1, step)
    
    # Detect fall in each window
    window_results = []
    fall_detections = []
    
    for window_idx, clip in enumerate(clips):
        # Detect fall in this window
        is_fall = detect_fall_in_keypoints(clip)
        is_fall = bool(is_fall)  # Ensure Python bool for JSON serialization
        fall_detections.append(1.0 if is_fall else 0.0)
        
        # Calculate window frame range
        window_start = window_idx * step
        window_end = window_start + clip_length - 1
        if window_end >= keypoints.shape[0]:
            window_end = keypoints.shape[0] - 1
        
        window_results.append({
            'window_num': int(window_idx),
            'window_start': int(window_start),
            'window_end': int(window_end),
            'is_fall': is_fall,  # Python bool
            'fall_probability': float(1.0 if is_fall else 0.0)
        })
    
    # Average fall probability across all windows
    avg_fall_prob = float(np.mean(fall_detections))
    is_fall_overall = bool(avg_fall_prob > 0.5)  # Convert to Python bool for JSON serialization
    
    result = {
        'is_fall': is_fall_overall,  # Python bool
        'fall_probability': avg_fall_prob,
        'num_windows': int(len(clips)),
        'windows_with_fall': int(np.sum(fall_detections))
    }
    
    if return_per_window:
        # Convert all booleans in window_results to Python bools
        for window_result in window_results:
            window_result['is_fall'] = bool(window_result['is_fall'])
            window_result['fall_probability'] = float(window_result['fall_probability'])
            window_result['window_num'] = int(window_result['window_num'])
            window_result['window_start'] = int(window_result['window_start'])
            window_result['window_end'] = int(window_result['window_end'])
        
        result['per_window'] = window_results
        result['clip_length'] = int(clip_length)
        result['overlap'] = float(overlap)
        result['step'] = int(step)
        result['keypoints'] = keypoints  # Return for visualization
    
    return result

