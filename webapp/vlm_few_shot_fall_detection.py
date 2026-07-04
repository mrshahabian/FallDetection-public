"""
Few-shot VLM fall detection using time windows.

Uses the same frozen CLIP backbone as the zero-shot mode, plus a lightweight
linear classifier trained on video-level CLIP embeddings (see
`evaluation/train_few_shot_vlm.py` and `vlm/few_shot_train.py`). The per-window
/ majority-vote pipeline mirrors `evaluation/evaluators.py::FewShotVLMEvaluator`.
"""

import os
import sys
import cv2
import torch
from typing import Dict, Optional
from PIL import Image

# Add paths
parent_dir = os.path.dirname(os.path.dirname(__file__))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.join(parent_dir, 'src'))

# Try to import VLM modules
try:
    import vlm.vlm_model as vlm_model_module
    from vlm.few_shot_train import load_classifier
    from evaluation.metrics import aggregate_window_results

    VisionLanguageModel = vlm_model_module.VisionLanguageModel

    VLM_AVAILABLE = True
    print("VLM modules loaded successfully for few-shot fall detection")
except ImportError as e:
    print(f"Warning: VLM modules not available (ImportError): {e}")
    import traceback
    traceback.print_exc()
    VLM_AVAILABLE = False
    VisionLanguageModel = None
    load_classifier = None
except Exception as e:
    print(f"Warning: VLM modules not available (Exception): {e}")
    import traceback
    traceback.print_exc()
    VLM_AVAILABLE = False
    VisionLanguageModel = None
    load_classifier = None

# Default checkpoint produced by evaluation/train_few_shot_vlm.py
DEFAULT_CLASSIFIER_PATH = os.path.join(parent_dir, 'evaluation', 'checkpoints', 'few_shot_vlm_classifier.pt')
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
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return Image.fromarray(frame_rgb)
        return None
    except Exception as e:
        print(f"Error sampling frame {frame_index}: {e}")
        return None


def classify_frame(frame: Image.Image, vlm: "VisionLanguageModel", classifier) -> Dict[str, float]:
    """
    Classify a single frame as fall / non-fall using the trained linear classifier
    on top of the frozen CLIP image embedding.
    """
    try:
        inputs = vlm.processor(images=frame, return_tensors="pt")
        pixel_values = inputs['pixel_values'].to(vlm.device)

        with torch.no_grad():
            image_embed = vlm.encode_images(pixel_values, normalize=True)
            logit = classifier(image_embed).squeeze(-1)
            fall_probability = torch.sigmoid(logit).item()

        is_fall = fall_probability >= FALL_THRESHOLD

        return {
            'fall_probability': float(fall_probability),
            'is_fall': bool(is_fall),
        }
    except Exception as e:
        print(f"Error classifying frame: {e}")
        return {
            'fall_probability': 0.0,
            'is_fall': False,
        }


def predict_vlm_few_shot_fall_detection(
    video_path: str,
    clip_length: int = 32,
    overlap: float = 0.5,
    classifier_path: Optional[str] = None,
    return_per_window: bool = False
) -> Dict:
    """
    Few-shot VLM fall detection using time windows.
    Samples a frame from each time window and classifies it with the trained
    linear probe on frozen CLIP embeddings.

    Args:
        video_path: Path to video file
        clip_length: Length of each time window in frames
        overlap: Overlap ratio between windows
        classifier_path: Path to the trained classifier checkpoint. Defaults to
            evaluation/checkpoints/few_shot_vlm_classifier.pt.
        return_per_window: If True, return per-window results

    Returns:
        Dictionary with fall detection results
    """
    if not VLM_AVAILABLE:
        raise RuntimeError("VLM modules not available. Please install transformers library.")

    classifier_path = classifier_path or DEFAULT_CLASSIFIER_PATH
    if not os.path.exists(classifier_path):
        raise RuntimeError(
            f"Few-shot classifier checkpoint not found at {classifier_path}. "
            f"Train one with: python evaluation/train_few_shot_vlm.py"
        )

    # Load VLM model
    print("Loading VLM model for few-shot fall detection...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.empty_cache()

    try:
        vlm = VisionLanguageModel(
            model_name="openai/clip-vit-base-patch32",
            device=device
        )
    except Exception as e:
        print(f"Failed to load VLM model: {e}")
        if device == "cuda":
            print("Attempting to load on CPU...")
            torch.cuda.empty_cache()
            vlm = VisionLanguageModel(
                model_name="openai/clip-vit-base-patch32",
                device="cpu"
            )

    # Load the trained linear classifier on top of the frozen CLIP embedding
    print(f"Loading few-shot classifier from {classifier_path}...")
    classifier = load_classifier(classifier_path, input_dim=vlm.image_embed_dim, hidden_dim=None)
    classifier = classifier.to(vlm.device)
    classifier.eval()

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

    # Calculate time windows (same as skeleton models and zero-shot VLM)
    step = int(clip_length * (1 - overlap))
    step = max(1, step)

    windows = []
    window_idx = 0
    start = 0

    while start + clip_length <= total_frames:
        window_end = start + clip_length - 1
        windows.append({
            'window_num': window_idx,
            'window_start': int(start),
            'window_end': int(window_end),
            'center_frame': int(start + clip_length // 2)
        })
        start += step
        window_idx += 1

    if start < total_frames:
        windows.append({
            'window_num': window_idx,
            'window_start': int(max(0, total_frames - clip_length)),
            'window_end': int(total_frames - 1),
            'center_frame': int(total_frames - clip_length // 2)
        })

    print(f"Analyzing {len(windows)} time windows for few-shot fall detection...")

    window_results = []

    for window in windows:
        frame = sample_frame_from_window(video_path, window['center_frame'])

        if frame is not None:
            fall_result = classify_frame(frame, vlm, classifier)
            window_results.append({
                'window_num': window['window_num'],
                'window_start': window['window_start'],
                'window_end': window['window_end'],
                'frame_num': window['center_frame'],
                'is_fall': fall_result['is_fall'],
                'fall_probability': fall_result['fall_probability']
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
        'method': 'few_shot_vlm',
    }

    if return_per_window:
        result['per_window'] = window_results
        result['clip_length'] = clip_length
        result['overlap'] = overlap
        result['step'] = step
        result['keypoints'] = keypoints
        result['fps'] = float(fps) if fps > 0 else 30.0
        result['total_frames'] = int(total_frames)

    return result
