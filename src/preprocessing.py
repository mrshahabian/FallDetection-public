"""
Preprocessing pipeline for skeleton keypoint sequences
"""

import os
import numpy as np
from typing import List, Tuple, Optional
from pathlib import Path
import tqdm
import cv2


def generate_clips(keypoints: np.ndarray, clip_length: int = 32, 
                   overlap: float = 0.5) -> List[np.ndarray]:
    """
    Generate fixed-length clips from a keypoint sequence using sliding window.
    
    Args:
        keypoints: Array of shape (T, J, C) where T=total frames, J=joints, C=coords
        clip_length: Desired length of each clip
        overlap: Overlap ratio between consecutive clips (0.0 to 1.0)
        
    Returns:
        List of clip arrays, each of shape (clip_length, J, C)
    """
    T, J, C = keypoints.shape
    
    if T < clip_length:
        # Pad sequence if shorter than clip_length
        padding = np.zeros((clip_length - T, J, C))
        padded_keypoints = np.concatenate([keypoints, padding], axis=0)
        return [padded_keypoints]
    
    clips = []
    step = int(clip_length * (1 - overlap))
    step = max(1, step)  # Ensure at least 1 frame step
    
    start = 0
    while start + clip_length <= T:
        clip = keypoints[start:start + clip_length]
        clips.append(clip)
        start += step
    
    # Include the last clip if there's remaining frames
    if start < T:
        last_clip = keypoints[-clip_length:]
        if len(clips) == 0 or not np.array_equal(clips[-1], last_clip):
            clips.append(last_clip)
    
    return clips


def create_2d_image_representation(keypoints: np.ndarray) -> np.ndarray:
    """
    Create 2D image representation for 2D CNN.
    Format: T × (2J) where left half (columns 0:J) are x coords, right half (J:2J) are y coords.
    
    Args:
        keypoints: Array of shape (T, J, C) where C=2 (x, y)
        
    Returns:
        Array of shape (T, 2J) representing the 2D image
    """
    T, J, C = keypoints.shape
    assert C == 2, "Expected 2 coordinates (x, y)"
    
    # Separate x and y coordinates
    x_coords = keypoints[:, :, 0]  # Shape: (T, J)
    y_coords = keypoints[:, :, 1]  # Shape: (T, J)
    
    # Concatenate: left half = x, right half = y
    image_2d = np.concatenate([x_coords, y_coords], axis=1)  # Shape: (T, 2J)
    
    return image_2d


def create_pose_heatmap(keypoints: np.ndarray, image_size: Tuple[int, int] = (64, 64),
                        sigma: float = 2.0) -> np.ndarray:
    """
    Create pose heatmap visualization for debugging.
    
    Args:
        keypoints: Array of shape (T, J, C) normalized to [0, 1]
        image_size: Output heatmap size (H, W)
        sigma: Gaussian kernel sigma for keypoint rendering
        
    Returns:
        Array of shape (T, H, W) representing heatmaps for each frame
    """
    T, J, C = keypoints.shape
    H, W = image_size
    
    heatmaps = []
    
    for t in range(T):
        heatmap = np.zeros((H, W))
        
        for j in range(J):
            x, y = keypoints[t, j, 0], keypoints[t, j, 1]
            
            # Convert normalized coords to pixel coords
            px = int(x * W)
            py = int(y * H)
            
            # Clamp to image bounds
            px = max(0, min(W - 1, px))
            py = max(0, min(H - 1, py))
            
            # Create Gaussian around keypoint
            y_coords, x_coords = np.ogrid[:H, :W]
            gaussian = np.exp(-((x_coords - px)**2 + (y_coords - py)**2) / (2 * sigma**2))
            heatmap = np.maximum(heatmap, gaussian)
        
        heatmaps.append(heatmap)
    
    return np.array(heatmaps)


def preprocess_kth_skeletons(input_root: str, output_root: str,
                            clip_length: int = 32, overlap: float = 0.5,
                            save_heatmaps: bool = True) -> None:
    """
    Preprocess extracted skeleton keypoints into training clips.
    
    Args:
        input_root: Root directory containing extracted skeletons (e.g., ./data/openpose)
        output_root: Root directory to save preprocessed clips
        clip_length: Length of each clip in frames
        overlap: Overlap ratio for sliding window
        save_heatmaps: Whether to save pose heatmap visualizations
    """
    actions = ['walking', 'jogging', 'running', 'boxing', 'handwaving', 'handclapping']
    action_to_label = {
        'walking': 0, 'jogging': 1, 'running': 2,
        'boxing': 3, 'handwaving': 4, 'handclapping': 5
    }
    
    os.makedirs(output_root, exist_ok=True)
    
    all_clips = []
    all_labels = []
    all_metadata = []
    
    # Process each action
    for action in actions:
        action_dir = os.path.join(input_root, action)
        if not os.path.exists(action_dir):
            print(f"Warning: Action directory not found: {action_dir}")
            continue
        
        label = action_to_label[action]
        
        # Process each skeleton file (support both .npy and .npz formats)
        skeleton_files = [f for f in os.listdir(action_dir) if f.endswith(('.npy', '.npz'))]
        
        for skeleton_file in tqdm.tqdm(skeleton_files, desc=f"Processing {action}"):
            skeleton_path = os.path.join(action_dir, skeleton_file)
            data = np.load(skeleton_path)
            
            # Handle both old .npy format (just keypoints) and new .npz format (keypoints + bboxes)
            if skeleton_file.endswith('.npz') and 'keypoints' in data:
                keypoints = data['keypoints']  # Shape: (T, J, C)
                bboxes = data.get('bboxes', None)  # Shape: (T, 4) or None
            else:
                # Old .npy format or .npz with 'skeleton' key (from preprocessing)
                if 'skeleton' in data:
                    keypoints = data['skeleton']
                else:
                    # Direct array in .npy file
                    keypoints = data
                bboxes = None
            
            # Generate clips
            clips = generate_clips(keypoints, clip_length, overlap)
            
            # Generate corresponding bounding box clips if available
            bbox_clips = None
            if bboxes is not None:
                # Reshape bboxes to (T, 1, 4) to match generate_clips expected format, then reshape back
                bboxes_reshaped = bboxes.reshape(bboxes.shape[0], 1, bboxes.shape[1])  # (T, 1, 4)
                bbox_clips_reshaped = generate_clips(bboxes_reshaped, clip_length, overlap)
                # Reshape back to (clip_length, 4)
                bbox_clips = [clip.reshape(clip_length, 4) for clip in bbox_clips_reshaped]
            
            # Save each clip
            video_name = os.path.splitext(skeleton_file)[0]
            
            for clip_idx, clip in enumerate(clips):
                # Create 2D image representation
                image_2d = create_2d_image_representation(clip)  # Shape: (T, 2J)
                
                # Create heatmap if requested
                heatmap = None
                if save_heatmaps:
                    heatmap = create_pose_heatmap(clip)  # Shape: (T, H, W)
                
                # Save clip as .npz
                clip_filename = f"{video_name}_clip{clip_idx:03d}.npz"
                clip_path = os.path.join(output_root, clip_filename)
                
                # Get corresponding bounding box clip if available
                clip_bboxes = bbox_clips[clip_idx] if bbox_clips is not None else None
                
                np.savez(
                    clip_path,
                    skeleton=clip,  # Original (T, J, C)
                    image_2d=image_2d,  # 2D representation (T, 2J)
                    heatmap=heatmap,  # Optional heatmap (T, H, W)
                    bboxes=clip_bboxes,  # Bounding boxes for this clip (T, 4) or None
                    label=label,
                    video_name=video_name,
                    clip_index=clip_idx,
                    action=action
                )
                
                all_clips.append(clip)
                all_labels.append(label)
                all_metadata.append({
                    'video_name': video_name,
                    'clip_index': clip_idx,
                    'action': action
                })
    
    print(f"Generated {len(all_clips)} clips from {input_root}")
    print(f"Saved to {output_root}")


def split_dataset(data_root: str, split_strategy: str = "random",
                 train_ratio: float = 0.7, val_ratio: float = 0.15,
                 test_ratio: float = 0.15, seed: int = 42) -> Tuple[List[str], List[str], List[str]]:
    """
    Split dataset into train/val/test sets.
    
    Args:
        data_root: Root directory containing preprocessed clips
        split_strategy: 'random' or 'standard' (person-based for KTH)
        train_ratio: Training set ratio (for random split)
        val_ratio: Validation set ratio (for random split)
        test_ratio: Test set ratio (for random split)
        seed: Random seed
        
    Returns:
        Tuple of (train_files, val_files, test_files) - lists of file paths
    """
    np.random.seed(seed)
    
    # Get all clip files
    clip_files = [f for f in os.listdir(data_root) if f.endswith('.npz')]
    clip_files = [os.path.join(data_root, f) for f in clip_files]
    
    if split_strategy == "random":
        # Random split
        np.random.shuffle(clip_files)
        n_total = len(clip_files)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        train_files = clip_files[:n_train]
        val_files = clip_files[n_train:n_train + n_val]
        test_files = clip_files[n_train + n_val:]
        
    elif split_strategy == "standard":
        # Standard KTH split: persons 1-16 train, 17-25 test
        # KTH video naming: personXX_scenario_action.avi
        train_files = []
        val_files = []
        test_files = []
        
        train_persons = set(range(1, 17))  # Persons 1-16
        test_persons = set(range(17, 26))  # Persons 17-25
        
        for clip_file in clip_files:
            # Extract person number from filename
            filename = os.path.basename(clip_file)
            # Format: personXX_action_clipXXX.npz
            parts = filename.split('_')
            if len(parts) >= 2 and parts[0].startswith('person'):
                try:
                    person_num = int(parts[0].replace('person', ''))
                    if person_num in train_persons:
                        train_files.append(clip_file)
                    elif person_num in test_persons:
                        test_files.append(clip_file)
                    else:
                        # Unknown person, add to train
                        train_files.append(clip_file)
                except ValueError:
                    # Can't parse person number, add to train
                    train_files.append(clip_file)
            else:
                # Unknown format, add to train
                train_files.append(clip_file)
        
        # Split train into train/val (80/20 of train set)
        np.random.shuffle(train_files)
        n_train = int(len(train_files) * 0.8)
        val_files = train_files[n_train:]
        train_files = train_files[:n_train]
    
    else:
        raise ValueError(f"Unknown split strategy: {split_strategy}")
    
    print(f"Split: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")
    
    return train_files, val_files, test_files


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        input_root = sys.argv[1]
        output_root = sys.argv[2] if len(sys.argv) > 2 else "./data/preprocessed"
        clip_length = int(sys.argv[3]) if len(sys.argv) > 3 else 32
        overlap = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5
        
        preprocess_kth_skeletons(input_root, output_root, clip_length, overlap)

