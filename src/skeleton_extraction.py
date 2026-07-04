"""
Skeleton extraction from videos using OpenPose and YOLOv11-pose
"""

import os
import sys
import time
import cv2
import numpy as np
from typing import Tuple, Optional, List
from pathlib import Path
import tqdm

# COCO 17 keypoint format mapping
COCO_17_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

# Mapping from OpenPose BODY_25 to COCO 17 format
# OpenPose BODY_25 has 25 keypoints, we need to map to COCO 17
OPENPOSE_TO_COCO_17 = {
    0: 0,   # Nose
    1: 2,   # Neck -> (not in COCO, use midpoint of shoulders)
    2: 6,   # Right shoulder
    3: 7,   # Right elbow
    4: 8,   # Right wrist
    5: 5,   # Left shoulder
    6: 6,   # Left elbow (mapped to right elbow index, will be corrected)
    7: 7,   # Left wrist
    8: 12,  # Mid-hip -> Right hip
    9: 12,  # Right hip
    10: 11, # Left hip
    11: 14, # Right knee
    12: 13, # Left knee
    13: 16, # Right ankle
    14: 15, # Left ankle
    15: 1,  # Right eye
    16: 0,  # Left eye
    17: 3,  # Right ear
    18: 4,  # Left ear
}

# Better mapping: OpenPose BODY_25 indices to COCO 17
# OpenPose: 0=nose, 1=neck, 2-4=right arm, 5-7=left arm, 8-10=hip, 11-14=legs, 15-18=face
# COCO 17: 0=nose, 1-2=eyes, 3-4=ears, 5-6=shoulders, 7-8=elbows, 9-10=wrists, 11-12=hips, 13-14=knees, 15-16=ankles
OPENPOSE_BODY25_TO_COCO17 = [
    0,      # 0: nose -> 0: nose
    15,     # 1: neck -> 1: left_eye (approximate)
    16,     # 2: right_shoulder -> 2: right_eye (approximate)
    2,      # 3: right_shoulder -> 3: left_ear
    5,      # 4: left_shoulder -> 4: right_ear
    2,      # 5: right_shoulder -> 5: left_shoulder
    5,      # 6: left_shoulder -> 6: right_shoulder
    3,      # 7: right_elbow -> 7: left_elbow
    6,      # 8: left_elbow -> 8: right_elbow
    4,      # 9: right_wrist -> 9: left_wrist
    7,      # 10: left_wrist -> 10: right_wrist
    9,      # 11: right_hip -> 11: left_hip
    12,     # 12: left_hip -> 12: right_hip
    11,     # 13: right_knee -> 13: left_knee
    14,     # 14: left_knee -> 14: right_knee
    13,     # 15: right_ankle -> 15: left_ankle
    14,     # 16: left_ankle -> 16: right_ankle
]


def extract_with_openpose_python_api(video_path: str, op_module, 
                                     output_path: Optional[str] = None,
                                     model_path: Optional[str] = None) -> np.ndarray:
    """
    Extract skeleton using OpenPose Python API.
    
    Args:
        video_path: Path to input video file
        op_module: Imported pyopenpose module
        output_path: Optional path to save keypoints
        model_path: Path to OpenPose models directory
        
    Returns:
        numpy array of shape (T, J, C) where T=frames, J=17 joints, C=2 (x, y)
    """
    import pyopenpose as op
    
    # Configure OpenPose
    params = dict()
    params["model_folder"] = model_path if model_path else os.environ.get('OPENPOSE_MODELS', 'models/')
    params["model_pose"] = "BODY_25"  # Use BODY_25, will convert to COCO 17
    params["number_people_max"] = 1  # Only detect one person
    
    # Initialize OpenPose
    opWrapper = op.Wrapper()
    opWrapper.configure(params)
    opWrapper.start()
    
    # Process video
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    keypoints_list = []
    
    print(f"Processing {total_frames} frames...")
    
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        datum = op.Datum()
        datum.cvInputData = frame
        opWrapper.emplaceAndPop(op.VectorDatum([datum]))
        
        # Extract keypoints
        if datum.poseKeypoints is not None and len(datum.poseKeypoints) > 0:
            # Get first person's keypoints (BODY_25 format: 25 keypoints)
            pose_keypoints = datum.poseKeypoints[0]  # Shape: (25, 3) - x, y, confidence
            
            # Convert BODY_25 to COCO 17
            coco_keypoints = np.zeros((17, 2))
            
            # Mapping BODY_25 to COCO 17
            # COCO: 0=nose, 1=left_eye, 2=right_eye, 3=left_ear, 4=right_ear,
            #       5=left_shoulder, 6=right_shoulder, 7=left_elbow, 8=right_elbow,
            #       9=left_wrist, 10=right_wrist, 11=left_hip, 12=right_hip,
            #       13=left_knee, 14=right_knee, 15=left_ankle, 16=right_ankle
            
            # OpenPose BODY_25: 0=nose, 1=neck, 2-4=right arm, 5-7=left arm,
            #                   8-10=hip, 11-14=legs, 15-18=face, 19-24=hand/feet
            
            mapping = {
                0: 0,   # nose -> nose
                15: 1,  # left eye -> left eye
                16: 2,  # right eye -> right eye
                17: 3,  # left ear -> left ear
                18: 4,  # right ear -> right ear
                5: 5,   # left shoulder -> left shoulder
                2: 6,   # right shoulder -> right shoulder
                6: 7,   # left elbow -> left elbow
                3: 8,   # right elbow -> right elbow
                7: 9,   # left wrist -> left wrist
                4: 10,  # right wrist -> right wrist
                10: 11, # left hip -> left hip
                9: 12,  # right hip -> right hip
                12: 13, # left knee -> left knee
                11: 14, # right knee -> right knee
                14: 15, # left ankle -> left ankle
                13: 16, # right ankle -> right ankle
            }
            
            for coco_idx, body25_idx in mapping.items():
                if body25_idx < len(pose_keypoints):
                    x, y, conf = pose_keypoints[body25_idx]
                    if conf > 0.1:  # Confidence threshold
                        coco_keypoints[coco_idx, 0] = x
                        coco_keypoints[coco_idx, 1] = y
        else:
            # No person detected
            coco_keypoints = np.zeros((17, 2))
        
        keypoints_list.append(coco_keypoints)
        frame_idx += 1
        if frame_idx % 10 == 0:
            print(f"  Processed {frame_idx}/{total_frames} frames...", end='\r')
    
    cap.release()
    
    # Convert to numpy array: (T, J, C)
    keypoints_array = np.array(keypoints_list)
    
    if output_path:
        np.save(output_path, keypoints_array)
    
    return keypoints_array


def extract_with_openpose_cli(video_path: str, openpose_binary: str,
                              output_path: Optional[str] = None,
                              net_resolution: str = "-1x160",
                              scale_number: int = 1,
                              scale_gap: float = 0.3) -> np.ndarray:
    """
    Extract skeleton using OpenPose command-line interface.
    
    Args:
        video_path: Path to input video file (must be absolute)
        openpose_binary: Path to OpenPose binary
        output_path: Optional path to save keypoints
        net_resolution: Network resolution (e.g., "-1x160" for speed, "-1x320" for accuracy)
        scale_number: Number of scales (1 is fastest, higher is more accurate but slower)
        scale_gap: Scale gap parameter (0.3 is a good default)
        
    Returns:
        numpy array of shape (T, J, C) where T=frames, J=17 joints, C=2 (x, y)
    """
    import subprocess
    import json
    import tempfile
    
    # OpenPose must be run from its root directory to find models
    # Binary is at: ~/openpose/build/examples/openpose/openpose.bin
    # Root is: ~/openpose (where models/ directory is)
    # Find OpenPose root directory
    if 'build/examples/openpose' in openpose_binary:
        # Standard structure: build/examples/openpose/openpose.bin
        openpose_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(openpose_binary))))
    elif 'build' in openpose_binary:
        # Alternative: build/.../openpose.bin
        parts = openpose_binary.split('/')
        build_idx = [i for i, p in enumerate(parts) if p == 'build'][0]
        openpose_root = '/'.join(parts[:build_idx])
    else:
        # Fallback: assume parent of parent
        openpose_root = os.path.dirname(os.path.dirname(openpose_binary))
    
    # Verify models directory exists
    models_dir = os.path.join(openpose_root, 'models')
    if not os.path.exists(models_dir):
        # Try common alternative locations
        alt_roots = [
            os.path.expanduser('~/openpose'),
            '/home/reza/openpose',
            os.path.dirname(os.path.dirname(os.path.dirname(openpose_binary)))
        ]
        for alt_root in alt_roots:
            if os.path.exists(os.path.join(alt_root, 'models')):
                openpose_root = alt_root
                models_dir = os.path.join(openpose_root, 'models')
                break
        else:
            raise RuntimeError(f"OpenPose models directory not found. Tried: {models_dir}")
    
    # Verify we found the right root
    if not os.path.exists(os.path.join(openpose_root, 'models', 'pose')):
        raise RuntimeError(f"OpenPose models/pose directory not found in {openpose_root}")
    
    # Convert video path to absolute
    video_path = os.path.abspath(video_path)
    
    # Create temporary directory for OpenPose output (use absolute path)
    temp_dir = tempfile.mkdtemp()
    json_output_dir = os.path.join(temp_dir, 'json_output')
    os.makedirs(json_output_dir, exist_ok=True)
    
    try:
        # Run OpenPose from its root directory
        # Use absolute path to binary
        # Performance settings: net_resolution="-1x160", scale_number=1, scale_gap=0.3
        # These optimize for speed on low-end devices
        cmd = [
            openpose_binary,
            '--video', video_path,
            '--write_json', json_output_dir,  # Use absolute path for output
            '--display', '0',
            '--render_pose', '0',
            '--model_pose', 'BODY_25',
            '--number_people_max', '1',  # Only detect one person
            '--net_resolution', net_resolution,  # Lower resolution for speed: "-1x160"
            '--scale_number', str(scale_number),  # Single scale (faster)
            '--scale_gap', str(scale_gap)  # Scale gap parameter
        ]
        
        print(f"Running OpenPose with performance optimizations...")
        print(f"  Resolution: {net_resolution}, Scale: {scale_number}")
        print(f"  Command: {' '.join(cmd)}")
        print(f"  Working directory: {openpose_root}")
        print(f"  Processing video (this may take several minutes)...")
        sys.stdout.flush()  # Ensure output is shown
        
        # Run from OpenPose root directory (critical for finding models)
        # CPU-only mode is slow, use longer timeout (30 minutes for a video)
        # Use Popen to show output in real-time and provide progress feedback
        print(f"  Starting OpenPose process...")
        sys.stdout.flush()
        
        process = subprocess.Popen(
            cmd,
            cwd=openpose_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,  # Line buffered
            universal_newlines=True
        )
        
        # Stream output in real-time and show progress
        output_lines = []
        start_time = time.time()
        last_progress_time = start_time
        progress_interval = 10.0  # Show progress every 10 seconds
        
        print(f"  OpenPose is processing (this may take several minutes)...")
        sys.stdout.flush()
        
        # Read output line by line
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                output_lines.append(output.strip())
                # Show important messages immediately
                if any(keyword in output.lower() for keyword in ['error', 'warning', 'failed', 'exception']):
                    print(f"  ⚠ OpenPose: {output.strip()}")
                    sys.stdout.flush()
            
            # Show progress indicator every few seconds
            current_time = time.time()
            if current_time - last_progress_time >= progress_interval:
                elapsed = int(current_time - start_time)
                print(f"  ⏳ Still processing... (elapsed: {elapsed}s)", end='\r')
                sys.stdout.flush()
                last_progress_time = current_time
        
        # Wait for process to finish and get return code
        # Don't use communicate() here since we've already read from stdout
        return_code = process.wait()
        
        # Clear progress line
        print()  # New line after progress indicator
        
        result = type('obj', (object,), {
            'returncode': return_code,
            'stdout': '\n'.join(output_lines),
            'stderr': ''  # stderr is redirected to stdout
        })()
        
        if result.returncode != 0:
            error_msg = result.stderr if result.stderr else result.stdout
            print(f"\n✗ OpenPose failed with return code {result.returncode}")
            if error_msg:
                print(f"Error output:\n{error_msg[:500]}")  # Show first 500 chars
            raise RuntimeError(f"OpenPose failed (return code {result.returncode}): {error_msg[:200]}")
        
        print(f"OpenPose completed. Reading JSON files...")
        
        # Read JSON files
        json_files = sorted([f for f in os.listdir(json_output_dir) if f.endswith('.json')])
        print(f"Found {len(json_files)} JSON files")
        keypoints_list = []
        
        for idx, json_file in enumerate(json_files):
            json_path = os.path.join(json_output_dir, json_file)
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            if (idx + 1) % 10 == 0:
                print(f"  Processed {idx + 1}/{len(json_files)} JSON files...", end='\r')
            
            # Extract keypoints (same mapping as Python API)
            if 'people' in data and len(data['people']) > 0:
                person = data['people'][0]
                pose_keypoints = np.array(person['pose_keypoints_2d']).reshape(-1, 3)  # (25, 3)
                
                # Convert to COCO 17 (same mapping as Python API)
                coco_keypoints = np.zeros((17, 2))
                mapping = {
                    0: 0, 15: 1, 16: 2, 17: 3, 18: 4,
                    5: 5, 2: 6, 6: 7, 3: 8, 7: 9, 4: 10,
                    10: 11, 9: 12, 12: 13, 11: 14, 14: 15, 13: 16
                }
                
                for coco_idx, body25_idx in mapping.items():
                    if body25_idx < len(pose_keypoints):
                        x, y, conf = pose_keypoints[body25_idx]
                        if conf > 0.1:
                            coco_keypoints[coco_idx, 0] = x
                            coco_keypoints[coco_idx, 1] = y
            else:
                coco_keypoints = np.zeros((17, 2))
            
            keypoints_list.append(coco_keypoints)
            
            if (idx + 1) % 10 == 0:
                print(f"  Processed {idx + 1}/{len(json_files)} JSON files...", end='\r')
                sys.stdout.flush()
        
        print(f"\n  ✓ Processed {len(keypoints_list)} frames")
        sys.stdout.flush()
        
        # Convert to numpy array
        keypoints_array = np.array(keypoints_list)
        
        if output_path:
            np.save(output_path, keypoints_array)
        
        return keypoints_array
        
    finally:
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


def extract_with_openpose(video_path: str, output_path: Optional[str] = None,
                         model_path: Optional[str] = None,
                         openpose_binary: Optional[str] = None,
                         net_resolution: str = "-1x160",
                         scale_number: int = 1,
                         scale_gap: float = 0.3) -> np.ndarray:
    """
    Extract skeleton keypoints from video using OpenPose.
    Tries multiple methods: Python API -> Command Line -> Placeholder
    
    Args:
        video_path: Path to input video file
        output_path: Optional path to save keypoints
        model_path: Path to OpenPose models directory (for Python API)
        openpose_binary: Path to OpenPose binary (for CLI)
        net_resolution: Network resolution (e.g., "-1x160" for speed, "-1x320" for accuracy)
        scale_number: Number of scales (1 is fastest, higher is more accurate but slower)
        scale_gap: Scale gap parameter (0.3 is a good default)
        
    Returns:
        numpy array of shape (T, J, C) where T=frames, J=17 joints, C=2 (x, y)
    """
    # Try Python API first
    try:
        import pyopenpose as op
        print(f"Using OpenPose Python API for {video_path}")
        return extract_with_openpose_python_api(video_path, op, output_path, model_path)
    except ImportError:
        pass
    
    # Try command line
    if openpose_binary:
        if os.path.exists(openpose_binary) and os.access(openpose_binary, os.X_OK):
            try:
                print(f"Using OpenPose CLI for {video_path}")
                return extract_with_openpose_cli(
                    video_path, openpose_binary, output_path,
                    net_resolution=net_resolution,
                    scale_number=scale_number,
                    scale_gap=scale_gap
                )
            except Exception as e:
                print(f"OpenPose CLI failed: {e}")
    
    # Check PATH for openpose binary
    import subprocess
    result = subprocess.run(['which', 'openpose'], capture_output=True, text=True)
    if result.returncode == 0:
        openpose_path = result.stdout.strip()
        try:
            print(f"Using OpenPose CLI (from PATH) for {video_path}")
            return extract_with_openpose_cli(video_path, openpose_path, output_path)
        except Exception as e:
            print(f"OpenPose CLI failed: {e}")
    
    # Check default OpenPose installation location
    default_openpose_binary = os.path.expanduser("~/openpose/build/examples/openpose/openpose.bin")
    if os.path.exists(default_openpose_binary) and os.access(default_openpose_binary, os.X_OK):
        try:
            print(f"Using OpenPose binary at: {default_openpose_binary}")
            return extract_with_openpose_cli(
                video_path, default_openpose_binary, output_path,
                net_resolution=net_resolution,
                scale_number=scale_number,
                scale_gap=scale_gap
            )
        except Exception as e:
            print(f"OpenPose CLI failed: {e}")
    
    # Fallback to placeholder
    print(f"WARNING: OpenPose not found. Using placeholder extraction for {video_path}")
    print("To use actual OpenPose:")
    print("  1. Install OpenPose Python API: https://github.com/CMU-Perceptual-Computing-Lab/openpose")
    print("  2. Or install OpenPose binary and add to PATH")
    print("  3. Or specify openpose_binary path in config")
    
    # Placeholder implementation
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    cap.release()
    
    num_frames = len(frames)
    num_joints = 17
    num_coords = 2
    
    h, w = frames[0].shape[:2] if frames else (120, 160)
    keypoints = np.zeros((num_frames, num_joints, num_coords))
    
    center_x, center_y = w / 2, h / 2
    for t in range(num_frames):
        for j in range(num_joints):
            keypoints[t, j, 0] = center_x + np.random.normal(0, 10)
            keypoints[t, j, 1] = center_y + np.random.normal(0, 10)
    
    if output_path:
        np.save(output_path, keypoints)
    
    return keypoints


def extract_with_yolov11(video_path: str, output_path: Optional[str] = None,
                        model_name: str = "yolo11n-pose.pt",
                        confidence: float = 0.25) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract skeleton keypoints and bounding boxes from video using YOLOv11-pose.
    
    Args:
        video_path: Path to input video file
        output_path: Optional path to save keypoints and bounding boxes
        model_name: YOLOv11 model name (e.g., 'yolo11n-pose.pt')
        confidence: Confidence threshold for detection
        
    Returns:
        Tuple of:
        - keypoints: numpy array of shape (T, J, C) where T=frames, J=17 joints, C=2 (x, y)
        - bounding_boxes: numpy array of shape (T, 4) where each row is [x1, y1, x2, y2]
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("WARNING: ultralytics not installed. Using placeholder YOLOv11 extraction.")
        return _placeholder_yolo_extraction(video_path, output_path)
    
    # Load YOLOv11-pose model
    model = YOLO(model_name)
    
    # Process video
    cap = cv2.VideoCapture(video_path)
    keypoints_list = []
    bboxes_list = []
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run YOLOv11-pose inference
        results = model(frame, conf=confidence, verbose=False)
        
        # Extract keypoints and bounding box for the first (largest) person detected
        if (len(results) > 0 and 
            results[0].keypoints is not None and 
            len(results[0].keypoints.data) > 0 and
            len(results[0].boxes) > 0):
            # Extract keypoints
            keypoints = results[0].keypoints.data[0].cpu().numpy()  # Shape: (17, 3) - x, y, confidence
            keypoints_xy = keypoints[:, :2]  # Shape: (17, 2)
            
            # Extract bounding box (x1, y1, x2, y2)
            box = results[0].boxes[0].xyxy[0].cpu().numpy()  # Shape: (4,)
            bbox = box  # [x1, y1, x2, y2]
        else:
            # No person detected, use zeros
            keypoints_xy = np.zeros((17, 2))
            bbox = np.zeros(4)  # [x1, y1, x2, y2]
        
        keypoints_list.append(keypoints_xy)
        bboxes_list.append(bbox)
    
    cap.release()
    
    # Convert to numpy arrays
    keypoints_array = np.array(keypoints_list)  # (T, J, C)
    bboxes_array = np.array(bboxes_list)  # (T, 4)
    
    if output_path:
        # Save both keypoints and bounding boxes
        if output_path.endswith('.npy'):
            # Save as .npz to store multiple arrays
            output_path_npz = output_path.replace('.npy', '.npz')
        else:
            output_path_npz = output_path + '.npz' if not output_path.endswith('.npz') else output_path
        np.savez(output_path_npz, keypoints=keypoints_array, bboxes=bboxes_array)
    
    return keypoints_array, bboxes_array


def _placeholder_yolo_extraction(video_path: str, output_path: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Placeholder YOLOv11 extraction when ultralytics is not available."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    cap.release()
    
    num_frames = len(frames)
    num_joints = 17
    num_coords = 2
    
    h, w = frames[0].shape[:2] if frames else (120, 160)
    keypoints = np.zeros((num_frames, num_joints, num_coords))
    bboxes = np.zeros((num_frames, 4))  # [x1, y1, x2, y2]
    
    center_x, center_y = w / 2, h / 2
    bbox_w, bbox_h = w * 0.4, h * 0.6  # Approximate person bounding box
    for t in range(num_frames):
        for j in range(num_joints):
            keypoints[t, j, 0] = center_x + np.random.normal(0, 10)
            keypoints[t, j, 1] = center_y + np.random.normal(0, 10)
        # Placeholder bounding box
        bboxes[t] = [center_x - bbox_w/2, center_y - bbox_h/2, 
                     center_x + bbox_w/2, center_y + bbox_h/2]
    
    if output_path:
        # Save both keypoints and bounding boxes
        if output_path.endswith('.npy'):
            output_path_npz = output_path.replace('.npy', '.npz')
        else:
            output_path_npz = output_path + '.npz' if not output_path.endswith('.npz') else output_path
        np.savez(output_path_npz, keypoints=keypoints, bboxes=bboxes)
    
    print(f"WARNING: Using placeholder YOLOv11 extraction for {video_path}")
    print("Please install ultralytics: pip install ultralytics")
    
    return keypoints, bboxes


def normalize_keypoints(keypoints: np.ndarray, image_width: int, image_height: int,
                       center_on_hip: bool = True, hip_joint_index: int = 8) -> np.ndarray:
    """
    Normalize keypoints to [0, 1] range and optionally center on hip.
    
    Args:
        keypoints: Array of shape (T, J, C) or (J, C)
        image_width: Image width for scaling
        image_height: Image height for scaling
        center_on_hip: Whether to subtract hip joint coordinates
        hip_joint_index: Index of hip joint (COCO format: 8 is mid-hip, use average of 11,12)
        
    Returns:
        Normalized keypoints array
    """
    keypoints = keypoints.copy()
    
    # Center on hip if requested
    if center_on_hip:
        # For COCO 17, use average of left_hip (11) and right_hip (12) as reference
        if keypoints.ndim == 3:  # (T, J, C)
            hip_coords = (keypoints[:, 11, :] + keypoints[:, 12, :]) / 2.0  # Average of both hips
            keypoints = keypoints - hip_coords[:, np.newaxis, :]
        else:  # (J, C)
            hip_coords = (keypoints[11, :] + keypoints[12, :]) / 2.0
            keypoints = keypoints - hip_coords
    
    # Scale to [0, 1] by image dimensions
    if keypoints.ndim == 3:
        keypoints[:, :, 0] = keypoints[:, :, 0] / image_width
        keypoints[:, :, 1] = keypoints[:, :, 1] / image_height
    else:
        keypoints[:, 0] = keypoints[:, 0] / image_width
        keypoints[:, 1] = keypoints[:, 1] / image_height
    
    return keypoints


def process_kth_dataset(dataset_root: str, output_root: str, 
                       extractor: str = "yolov11",
                       config: Optional[dict] = None) -> None:
    """
    Process entire KTH dataset and extract skeletons.
    
    Args:
        dataset_root: Root directory of KTH dataset
        output_root: Root directory to save extracted skeletons
        extractor: 'openpose' or 'yolov11'
        config: Configuration dictionary with extraction settings
    """
    if config is None:
        config = {
            'normalize': {'scale_to_01': True, 'center_on_hip': True},
            'image_width': 160,
            'image_height': 120,
            'yolov11': {'model_name': 'yolo11n-pose.pt', 'confidence_threshold': 0.25}
        }
    
    actions = ['walking', 'jogging', 'running', 'boxing', 'handwaving', 'handclapping']
    
    # Create output directory
    extractor_dir = os.path.join(output_root, extractor)
    os.makedirs(extractor_dir, exist_ok=True)
    
    # Find all video files
    video_files = []
    for action in actions:
        action_dir = os.path.join(dataset_root, action)
        if not os.path.exists(action_dir):
            print(f"Warning: Action directory not found: {action_dir}")
            continue
        
        for video_file in os.listdir(action_dir):
            if video_file.endswith(('.avi', '.mp4', '.mov')):
                video_path = os.path.join(action_dir, video_file)
                video_files.append((video_path, action, video_file))
    
    print(f"Found {len(video_files)} videos to process")
    
    # Process each video
    for video_path, action, video_file in tqdm.tqdm(video_files, desc=f"Extracting with {extractor}"):
        # Create output path
        action_output_dir = os.path.join(extractor_dir, action)
        os.makedirs(action_output_dir, exist_ok=True)
        
        output_file = os.path.splitext(video_file)[0] + '.npz'
        output_path = os.path.join(action_output_dir, output_file)
        
        # Skip if already processed
        if os.path.exists(output_path):
            continue
        
        # Extract keypoints and bounding boxes
        if extractor == 'openpose':
            # Get OpenPose settings from config
            openpose_cfg = config.get('extraction', {}).get('openpose', {}) if config else {}
            openpose_bin = openpose_cfg.get('binary_path', None)
            if openpose_bin:
                openpose_bin = os.path.expanduser(openpose_bin)
            model_path = openpose_cfg.get('model_path', None)
            if model_path:
                model_path = os.path.expanduser(model_path)
            
            # Get performance settings from config
            net_resolution = openpose_cfg.get('net_resolution', '-1x160')
            scale_number = openpose_cfg.get('scale_number', 1)
            scale_gap = openpose_cfg.get('scale_gap', 0.3)
            
            keypoints = extract_with_openpose(
                video_path, None, 
                model_path=model_path,
                openpose_binary=openpose_bin,
                net_resolution=net_resolution,
                scale_number=scale_number,
                scale_gap=scale_gap
            )
            # OpenPose doesn't return bounding boxes, create zeros
            bboxes = np.zeros((keypoints.shape[0], 4))
        elif extractor == 'yolov11':
            model_name = config.get('yolov11', {}).get('model_name', 'yolo11n-pose.pt') if config else 'yolo11n-pose.pt'
            confidence = config.get('yolov11', {}).get('confidence_threshold', 0.25) if config else 0.25
            keypoints, bboxes = extract_with_yolov11(video_path, None, model_name, confidence)
        else:
            raise ValueError(f"Unknown extractor: {extractor}")
        
        # Normalize if requested
        normalize_cfg = config.get('normalize', {}) if config else {}
        if normalize_cfg.get('scale_to_01', False):
            image_width = config.get('image_width', 160) if config else 160
            image_height = config.get('image_height', 120) if config else 120
            keypoints = normalize_keypoints(
                keypoints,
                image_width,
                image_height,
                normalize_cfg.get('center_on_hip', True)
            )
            # Normalize bounding boxes to [0, 1] as well
            bboxes[:, [0, 2]] /= image_width  # x coordinates
            bboxes[:, [1, 3]] /= image_height  # y coordinates
        
        # Save both keypoints and bounding boxes
        np.savez(output_path, keypoints=keypoints, bboxes=bboxes)


if __name__ == "__main__":
    # Example usage
    import sys
    from pathlib import Path
    
    if len(sys.argv) > 1:
        dataset_root = sys.argv[1]
        output_root = sys.argv[2] if len(sys.argv) > 2 else "./data"
        extractor = sys.argv[3] if len(sys.argv) > 3 else "yolov11"
        
        # Load config if available
        config = None
        config_path = Path(__file__).parent.parent / "configs" / "extraction_config.yaml"
        if config_path.exists():
            try:
                from .utils import load_config
                config = load_config(str(config_path))
            except ImportError:
                # If relative import fails, try absolute
                import sys
                sys.path.insert(0, str(Path(__file__).parent))
                from utils import load_config
                config = load_config(str(config_path))
        
        # If no config or OpenPose binary not in config, try default location
        if extractor == 'openpose' and config:
            openpose_cfg = config.get('extraction', {}).get('openpose', {})
            if not openpose_cfg.get('binary_path'):
                # Try default location
                default_binary = Path.home() / "openpose" / "build" / "examples" / "openpose" / "openpose.bin"
                if default_binary.exists():
                    if 'extraction' not in config:
                        config['extraction'] = {}
                    if 'openpose' not in config['extraction']:
                        config['extraction']['openpose'] = {}
                    config['extraction']['openpose']['binary_path'] = str(default_binary)
        
        process_kth_dataset(dataset_root, output_root, extractor, config=config)

