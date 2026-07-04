"""
Zero-shot VLM fall detection using time windows.

Samples a frame from each time window and scores it against the balanced
fall / non-fall prompt bank from `evaluation/prompts.py`, using the same
contrastive fall-vs-non-fall scoring as `evaluation/evaluators.py::ZeroShotVLMEvaluator`.
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

# Try to import VLM modules
try:
    # Import VLM modules as a package (handles relative imports)
    import vlm.vlm_model as vlm_model_module
    import vlm.video_utils as video_utils_module
    import vlm.config as vlm_config_module
    from evaluation.prompts import get_all_prompts, get_prompt_info
    from evaluation.metrics import aggregate_window_results

    VisionLanguageModel = vlm_model_module.VisionLanguageModel
    get_clip_inputs_from_video = video_utils_module.get_clip_inputs_from_video
    load_config = vlm_config_module.load_config

    VLM_AVAILABLE = True
    print("VLM modules loaded successfully for improved fall detection")
except ImportError as e:
    print(f"Warning: VLM modules not available (ImportError): {e}")
    import traceback
    traceback.print_exc()
    VLM_AVAILABLE = False
    VisionLanguageModel = None
    load_config = None
except Exception as e:
    print(f"Warning: VLM modules not available (Exception): {e}")
    import traceback
    traceback.print_exc()
    VLM_AVAILABLE = False
    VisionLanguageModel = None
    load_config = None

# Contrastive fall-vs-non-fall scoring parameters (match evaluation/evaluators.py)
CONTRASTIVE_TEMPERATURE = 0.1
FALL_THRESHOLD = 0.5


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


def detect_fall_in_frame(
    frame: Image.Image,
    vlm: "VisionLanguageModel",
    processor,
    fall_text_embeds: torch.Tensor,
    non_fall_text_embeds: torch.Tensor
) -> Dict[str, float]:
    """
    Detect fall in a single frame using the balanced fall / non-fall prompt bank.

    Uses the contrastive approach from evaluation/evaluators.py: average cosine
    similarity to fall prompts vs. non-fall prompts, then a sigmoid of the
    difference (sharpened with a low temperature) instead of a softmax over
    all prompts. This gives much better separation than treating fall
    detection as classification over ~200 prompts.

    Args:
        frame: PIL Image
        vlm: VisionLanguageModel instance
        processor: CLIP processor
        fall_text_embeds: Pre-encoded embeddings for the fall prompts
        non_fall_text_embeds: Pre-encoded embeddings for the non-fall prompts

    Returns:
        Dictionary with fall_probability and is_fall
    """
    try:
        # Process image
        inputs = processor(images=frame, return_tensors="pt")
        pixel_values = inputs['pixel_values'].to(vlm.device)

        # Encode image
        image_embed = vlm.encode_images(pixel_values, normalize=True)

        # Compute similarities to each prompt group
        fall_similarities = vlm.compute_similarity(image_embed, fall_text_embeds, temperature=1.0)
        non_fall_similarities = vlm.compute_similarity(image_embed, non_fall_text_embeds, temperature=1.0)

        # Average similarity across prompts within each group
        fall_sim = fall_similarities.mean(dim=0).mean().item()
        non_fall_sim = non_fall_similarities.mean(dim=0).mean().item()

        # Contrastive probability: sigmoid of the (fall - non_fall) similarity gap
        diff = fall_sim - non_fall_sim
        fall_probability = torch.sigmoid(torch.tensor(diff / CONTRASTIVE_TEMPERATURE)).item()

        is_fall = fall_probability >= FALL_THRESHOLD

        return {
            'fall_probability': float(fall_probability),
            'is_fall': bool(is_fall),
        }

    except Exception as e:
        print(f"Error detecting fall in frame: {e}")
        return {
            'fall_probability': 0.0,
            'is_fall': False,
        }


def predict_vlm_fall_detection_improved(
    video_path: str,
    clip_length: int = 32,
    overlap: float = 0.5,
    return_per_window: bool = False
) -> Dict:
    """
    Zero-shot VLM fall detection using time windows.
    Samples a frame from each time window and detects falls.

    Args:
        video_path: Path to video file
        clip_length: Length of each time window in frames
        overlap: Overlap ratio between windows
        return_per_window: If True, return per-window results

    Returns:
        Dictionary with fall detection results
    """
    if not VLM_AVAILABLE:
        raise RuntimeError("VLM modules not available. Please install transformers library.")

    # Load VLM model
    print("Loading VLM model for zero-shot fall detection...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.empty_cache()

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

    # Encode the balanced fall / non-fall prompt bank once and reuse it for every window
    prompts = get_all_prompts()
    prompt_info = get_prompt_info()
    num_fall = prompt_info['num_fall_prompts']
    fall_prompts = prompts[:num_fall]
    non_fall_prompts = prompts[num_fall:]

    print(f"Using {len(prompts)} balanced prompts ({len(fall_prompts)} fall, {len(non_fall_prompts)} non-fall)")
    fall_text_embeds = vlm.encode_texts(fall_prompts, normalize=True)
    non_fall_text_embeds = vlm.encode_texts(non_fall_prompts, normalize=True)

    # Extract skeleton keypoints for visualization
    from video_processor import extract_with_yolov11
    from config import YOLOV11_MODEL_NAME, YOLOV11_CONFIDENCE

    print("Extracting skeleton for visualization...")
    extraction_result = extract_with_yolov11(
        video_path=video_path,
        model_name=YOLOV11_MODEL_NAME,
        confidence=YOLOV11_CONFIDENCE
    )

    if isinstance(extraction_result, tuple):
        keypoints, bounding_boxes = extraction_result
    else:
        keypoints = extraction_result
        bounding_boxes = None

    if keypoints is None or (hasattr(keypoints, 'shape') and keypoints.shape[0] == 0):
        print("Warning: Could not extract skeleton, continuing without visualization")
        keypoints = None

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

    print(f"Analyzing {len(windows)} time windows for fall detection...")

    # Detect fall in each window
    window_results = []

    for window in windows:
        # Sample frame from center of window
        frame = sample_frame_from_window(video_path, window['center_frame'])

        if frame is not None:
            # Detect fall in frame
            fall_result = detect_fall_in_frame(frame, vlm, processor, fall_text_embeds, non_fall_text_embeds)
            fall_prob = fall_result['fall_probability']
            is_fall = fall_result['is_fall']

            window_results.append({
                'window_num': window['window_num'],
                'window_start': window['window_start'],
                'window_end': window['window_end'],
                'frame_num': window['center_frame'],
                'is_fall': bool(is_fall),
                'fall_probability': float(fall_prob)
            })
        else:
            window_results.append({
                'window_num': window['window_num'],
                'window_start': window['window_start'],
                'window_end': window['window_end'],
                'frame_num': window['center_frame'],
                'is_fall': False,
                'fall_probability': 0.0
            })

    # Video-level decision: majority vote over windows (same as evaluation/metrics.py)
    aggregated = aggregate_window_results(window_results, true_label=True)

    result = {
        'is_fall': bool(aggregated['is_fall']),
        'fall_probability': float(aggregated['fall_probability']),
        'num_windows': len(windows),
        'windows_with_fall': int(aggregated['num_fall_windows']),
        'method': 'zero_shot_vlm',
        'num_prompts': len(prompts),
    }

    if return_per_window:
        result['per_window'] = window_results
        result['clip_length'] = clip_length
        result['overlap'] = overlap
        result['step'] = step
        result['keypoints'] = keypoints  # Return for visualization
        result['fps'] = float(fps) if fps > 0 else 30.0
        result['total_frames'] = int(total_frames)

    return result
