"""
VLM-based video description using time windows
Samples a frame from each time window and creates an aggregated description
"""

import os
import sys
import cv2
import numpy as np
import torch
from typing import Dict, List, Optional
from PIL import Image

# Add paths
parent_dir = os.path.dirname(os.path.dirname(__file__))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.join(parent_dir, 'src'))

from preprocessing import generate_clips

# Try to import VLM modules
try:
    # Import VLM modules as a package (handles relative imports)
    import vlm.vlm_model as vlm_model_module
    import vlm.video_utils as video_utils_module
    
    VisionLanguageModel = vlm_model_module.VisionLanguageModel
    get_clip_inputs_from_video = video_utils_module.get_clip_inputs_from_video
    
    VLM_AVAILABLE = True
    print("VLM modules loaded successfully for video description")
except ImportError as e:
    print(f"Warning: VLM modules not available (ImportError): {e}")
    import traceback
    traceback.print_exc()
    VLM_AVAILABLE = False
    VisionLanguageModel = None
except Exception as e:
    print(f"Warning: VLM modules not available (Exception): {e}")
    import traceback
    traceback.print_exc()
    VLM_AVAILABLE = False
    VisionLanguageModel = None


def sample_frame_from_window(video_path: str, frame_index: int) -> Optional[Image.Image]:
    """
    Sample a single frame from video at given index.
    
    Args:
        video_path: Path to video file
        frame_index: Frame index to sample
        
    Returns:
        PIL Image or None if failed
    """
    try:
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        cap.release()
        
        if ret and frame is not None:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Convert to PIL Image
            return Image.fromarray(frame_rgb)
        return None
    except Exception as e:
        print(f"Error sampling frame {frame_index}: {e}")
        return None


def describe_frame_with_vlm(frame: Image.Image, vlm: VisionLanguageModel, processor) -> str:
    """
    Describe a single frame using VLM with general description prompts.
    
    Args:
        frame: PIL Image
        vlm: VisionLanguageModel instance
        processor: CLIP processor
        
    Returns:
        Description text
    """
    # General description prompts
    description_prompts = [
        "a person walking",
        "a person running",
        "a person standing",
        "a person sitting",
        "a person jumping",
        "a person waving hands",
        "a person clapping hands",
        "a person boxing",
        "a person falling down",
        "a person lying on the ground",
        "a person moving quickly",
        "a person moving slowly",
        "a person performing an action",
        "a person in motion",
        "a person stationary"
    ]
    
    try:
        # Process image
        inputs = processor(images=frame, return_tensors="pt")
        pixel_values = inputs['pixel_values'].to(vlm.device)
        
        # Encode image
        image_embed = vlm.encode_images(pixel_values, normalize=True)
        
        # Encode text prompts
        text_embeds = vlm.encode_texts(description_prompts, normalize=True)
        
        # Compute similarities
        similarities = vlm.compute_similarity(image_embed, text_embeds)
        similarities = similarities.cpu().numpy()[0]
        
        # Get top 3 most similar prompts
        top_indices = np.argsort(similarities)[-3:][::-1]
        top_prompts = [description_prompts[i] for i in top_indices]
        top_scores = [float(similarities[i]) for i in top_indices]
        
        # Create description from top prompts
        if top_scores[0] > 0.25:  # Threshold for confidence
            # Use the top prompt as primary description
            description = top_prompts[0]
            # Add secondary actions if significant
            if len(top_prompts) > 1 and top_scores[1] > 0.2:
                description += f" and {top_prompts[1]}"
        else:
            description = "a person performing an activity"
        
        return description
    
    except Exception as e:
        print(f"Error describing frame: {e}")
        return "a person in the video"


def predict_vlm_video_description(
    video_path: str,
    clip_length: int = 32,
    overlap: float = 0.5,
    return_per_window: bool = False
) -> Dict:
    """
    Generate video description by analyzing frames from each time window.
    
    Args:
        video_path: Path to video file
        clip_length: Length of each time window in frames
        overlap: Overlap ratio between windows
        return_per_window: If True, return per-window descriptions
        
    Returns:
        Dictionary with:
        - 'description': Overall aggregated description
        - 'per_window': List of per-window descriptions (if return_per_window=True)
    """
    if not VLM_AVAILABLE:
        raise RuntimeError("VLM modules not available. Please install transformers library.")
    
    # Load VLM model
    print("Loading VLM model for video description...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.empty_cache()  # Clear cache before loading
    
    try:
        vlm = VisionLanguageModel(
            model_name="openai/clip-vit-base-patch32",
            device=device
        )
        processor = vlm.processor
    except Exception as e:
        print(f"Failed to load VLM model: {e}")
        if device == "cuda":
            print("Attempting to load on CPU...")
            torch.cuda.empty_cache()
            vlm = VisionLanguageModel(
                model_name="openai/clip-vit-base-patch32",
                device="cpu"
            )
            processor = vlm.processor
    
    # Get video properties
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    if total_frames == 0:
        raise ValueError("Could not read video or video has no frames")
    
    # Calculate time windows (same as skeleton models)
    step = int(clip_length * (1 - overlap))
    step = max(1, step)
    
    # Generate window frame ranges
    windows = []
    window_idx = 0
    start = 0
    
    while start + clip_length <= total_frames:
        window_end = start + clip_length - 1
        windows.append({
            'window_num': window_idx,
            'window_start': int(start),
            'window_end': int(window_end),
            'center_frame': int(start + clip_length // 2)  # Sample from center of window
        })
        start += step
        window_idx += 1
    
    # Include last window if there are remaining frames
    if start < total_frames:
        windows.append({
            'window_num': window_idx,
            'window_start': int(max(0, total_frames - clip_length)),
            'window_end': int(total_frames - 1),
            'center_frame': int(total_frames - clip_length // 2)
        })
    
    print(f"Analyzing {len(windows)} time windows...")
    
    # Describe each window
    window_descriptions = []
    all_descriptions = []
    
    for window in windows:
        # Sample frame from center of window
        frame = sample_frame_from_window(video_path, window['center_frame'])
        
        if frame is not None:
            # Describe frame
            description = describe_frame_with_vlm(frame, vlm, processor)
            all_descriptions.append(description)
            
            window_descriptions.append({
                'window_num': window['window_num'],
                'window_start': window['window_start'],
                'window_end': window['window_end'],
                'frame_num': window['center_frame'],
                'description': description
            })
        else:
            window_descriptions.append({
                'window_num': window['window_num'],
                'window_start': window['window_start'],
                'window_end': window['window_end'],
                'frame_num': window['center_frame'],
                'description': "unable to analyze frame"
            })
    
    # Aggregate descriptions
    overall_description = aggregate_descriptions(all_descriptions)
    
    result = {
        'description': overall_description,
        'num_windows': len(windows),
        'fps': float(fps) if fps > 0 else 30.0,
        'total_frames': int(total_frames)
    }
    
    if return_per_window:
        result['per_window'] = window_descriptions
        result['clip_length'] = clip_length
        result['overlap'] = overlap
        result['step'] = step
    
    return result


def aggregate_descriptions(descriptions: List[str]) -> str:
    """
    Aggregate multiple frame descriptions into a meaningful overall description.
    
    Args:
        descriptions: List of frame descriptions
        
    Returns:
        Aggregated description string
    """
    if not descriptions:
        return "Unable to analyze video"
    
    # Count unique descriptions
    from collections import Counter
    desc_counts = Counter(descriptions)
    
    # Get most common activities
    most_common = desc_counts.most_common(3)
    
    # Build aggregated description
    if len(most_common) == 1:
        # Single dominant activity
        activity = most_common[0][0]
        count = most_common[0][1]
        if count == len(descriptions):
            return f"The video shows {activity} throughout."
        else:
            return f"The video primarily shows {activity}."
    
    elif len(most_common) >= 2:
        # Multiple activities
        primary = most_common[0][0]
        secondary = most_common[1][0]
        
        # Check if activities are similar
        if primary == secondary or primary in secondary or secondary in primary:
            return f"The video shows {primary}."
        
        # Different activities - describe sequence
        if most_common[0][1] > most_common[1][1] * 1.5:
            # Primary activity dominates
            return f"The video primarily shows {primary}, with some {secondary}."
        else:
            # Multiple activities
            activities = [desc[0] for desc in most_common[:3]]
            return f"The video shows various activities including {', '.join(activities)}."
    
    return "The video shows various human activities."

