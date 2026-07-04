"""
Video utilities for frame sampling and preprocessing for CLIP.

This module provides functions to:
- Load videos and sample frames uniformly
- Preprocess frames for CLIP model input
- Handle edge cases (short videos, various formats, etc.)
"""

import os
import cv2
import numpy as np
import torch
from typing import List, Optional, Tuple, Union
from PIL import Image
import logging

logger = logging.getLogger(__name__)


def load_and_sample_frames(
    video_path: str,
    num_frames: int = 8,
    sampling_strategy: str = "uniform"
) -> List[np.ndarray]:
    """
    Load a video and sample frames uniformly across its duration.
    
    Args:
        video_path: Path to video file.
        num_frames: Number of frames to sample.
        sampling_strategy: "uniform" (default) or "temporal" (more frames at start/end).
        
    Returns:
        List of frames as numpy arrays (H, W, 3) in BGR format.
        
    Raises:
        FileNotFoundError: If video file doesn't exist.
        ValueError: If video cannot be opened or has no frames.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if total_frames == 0:
        cap.release()
        raise ValueError(f"Video has no frames: {video_path}")
    
    # Handle short videos: if fewer frames than requested, sample all frames
    if total_frames <= num_frames:
        logger.warning(
            f"Video has only {total_frames} frames, sampling all frames "
            f"(requested {num_frames})"
        )
        frame_indices = list(range(total_frames))
    else:
        # Sample frame indices uniformly
        if sampling_strategy == "uniform":
            frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        elif sampling_strategy == "temporal":
            # More frames at start and end (useful for detecting events)
            # Use a weighted distribution
            t = np.linspace(0, 1, num_frames)
            # Weight function: higher at start and end
            weights = np.sin(np.pi * t) ** 2
            weights = weights / weights.sum()
            cumulative = np.cumsum(weights)
            frame_indices = (cumulative * (total_frames - 1)).astype(int)
            # Ensure unique indices
            frame_indices = np.unique(frame_indices)
            # If we lost some frames, add evenly spaced ones
            while len(frame_indices) < num_frames:
                additional = np.linspace(0, total_frames - 1, num_frames, dtype=int)
                frame_indices = np.unique(np.concatenate([frame_indices, additional]))[:num_frames]
        else:
            raise ValueError(f"Unknown sampling strategy: {sampling_strategy}")
    
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        
        if not ret:
            logger.warning(f"Could not read frame {idx} from video {video_path}")
            continue
        
        # Convert BGR to RGB for CLIP (OpenCV reads as BGR)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    
    cap.release()
    
    if len(frames) == 0:
        raise ValueError(f"Could not extract any frames from video: {video_path}")
    
    return frames


def preprocess_frames_for_clip(
    frames: List[np.ndarray],
    processor,
    target_size: Optional[Tuple[int, int]] = None
) -> torch.Tensor:
    """
    Preprocess frames for CLIP model input.
    
    Args:
        frames: List of frames as numpy arrays (H, W, 3) in RGB format.
        processor: CLIP processor from transformers (handles resizing, normalization, etc.).
        target_size: Optional (H, W) tuple for resizing. If None, uses processor defaults.
        
    Returns:
        Tensor of shape [num_frames, 3, H, W] ready for CLIP.
    """
    if len(frames) == 0:
        raise ValueError("Cannot preprocess empty frame list")
    
    # Convert numpy arrays to PIL Images
    pil_images = []
    for frame in frames:
        if isinstance(frame, np.ndarray):
            # Ensure uint8 and correct shape
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)
            pil_image = Image.fromarray(frame)
        elif isinstance(frame, Image.Image):
            pil_image = frame
        else:
            raise TypeError(f"Unsupported frame type: {type(frame)}")
        
        pil_images.append(pil_image)
    
    # Use processor to preprocess (handles resizing, normalization, tensor conversion)
    # The processor expects a list of PIL Images
    inputs = processor(images=pil_images, return_tensors="pt")
    
    # Extract pixel_values tensor: [num_frames, 3, H, W]
    pixel_values = inputs["pixel_values"]
    
    return pixel_values


def get_clip_inputs_from_video(
    video_path: str,
    num_frames: int = 8,
    processor=None,
    sampling_strategy: str = "uniform",
    target_size: Optional[Tuple[int, int]] = None
) -> torch.Tensor:
    """
    High-level function: load video, sample frames, and preprocess for CLIP.
    
    This combines load_and_sample_frames and preprocess_frames_for_clip into
    a single convenient function.
    
    Args:
        video_path: Path to video file.
        num_frames: Number of frames to sample.
        processor: CLIP processor from transformers. If None, must be provided separately.
        sampling_strategy: "uniform" (default) or "temporal".
        target_size: Optional (H, W) tuple for resizing.
        
    Returns:
        Tensor of shape [num_frames, 3, H, W] ready for CLIP.
        
    Raises:
        ValueError: If processor is None.
    """
    if processor is None:
        raise ValueError(
            "processor must be provided. "
            "Use: processor = AutoProcessor.from_pretrained(model_name)"
        )
    
    # Load and sample frames
    frames = load_and_sample_frames(video_path, num_frames, sampling_strategy)
    
    # Preprocess for CLIP
    clip_inputs = preprocess_frames_for_clip(frames, processor, target_size)
    
    return clip_inputs


def frames_to_tensor(frames: List[np.ndarray]) -> torch.Tensor:
    """
    Convert list of numpy frames to a batched tensor.
    
    This is a helper function for cases where you already have frames
    and just need to convert them to tensor format.
    
    Args:
        frames: List of frames as numpy arrays (H, W, 3) in RGB format.
        
    Returns:
        Tensor of shape [num_frames, 3, H, W].
    """
    if len(frames) == 0:
        raise ValueError("Cannot convert empty frame list")
    
    # Stack frames and convert to tensor
    # frames is list of (H, W, 3) -> stack to (num_frames, H, W, 3)
    frames_array = np.stack(frames, axis=0)
    
    # Convert to tensor and permute to (num_frames, 3, H, W)
    tensor = torch.from_numpy(frames_array).float()
    tensor = tensor.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
    
    # Normalize to [0, 1] if not already
    if tensor.max() > 1.0:
        tensor = tensor / 255.0
    
    return tensor

