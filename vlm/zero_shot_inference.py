"""
Zero-shot fall detection using CLIP.

This module implements the zero-shot inference pipeline for activity recognition
(particularly fall detection) using natural language prompts.
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Optional
import logging

from .vlm_model import VisionLanguageModel
from .video_utils import get_clip_inputs_from_video
from .config import get_num_frames, get_zero_shot_prompts, get_zero_shot_temperature, get_zero_shot_aggregation

logger = logging.getLogger(__name__)


def zero_shot_fall_detection(
    video_path: str,
    vlm: VisionLanguageModel,
    prompts: Optional[List[str]] = None,
    num_frames: Optional[int] = None,
    temperature: Optional[float] = None,
    aggregation: Optional[str] = None,
    config: Optional[Dict] = None
) -> Dict[str, float]:
    """
    Perform zero-shot fall detection on a video using text prompts.
    
    This function:
    1. Samples frames from the video
    2. Encodes frames using CLIP vision encoder
    3. Encodes text prompts using CLIP text encoder
    4. Computes similarities between frames and prompts
    5. Aggregates similarities across frames
    6. Converts to probabilities using softmax
    
    Args:
        video_path: Path to video file.
        vlm: Initialized VisionLanguageModel instance.
        prompts: List of text prompts. If None, uses prompts from config.
        num_frames: Number of frames to sample. If None, uses config default.
        temperature: Temperature for softmax. If None, uses config default.
        aggregation: Aggregation method ("mean" or "max"). If None, uses config default.
        config: Configuration dictionary. Used if other parameters are None.
    
    Returns:
        Dictionary mapping each prompt to its probability.
        Also includes a 'fall_probability' key if prompts contain fall-related prompts.
    """
    # Get parameters from config if not provided
    if config is not None:
        if prompts is None:
            prompts = get_zero_shot_prompts(config)
        if num_frames is None:
            num_frames = get_num_frames(config)
        if temperature is None:
            temperature = get_zero_shot_temperature(config)
        if aggregation is None:
            aggregation = get_zero_shot_aggregation(config)
    
    # Default values if still None
    if prompts is None:
        prompts = [
            "a CCTV video of an elderly person falling from a bed",
            "an elderly person lying calmly in bed",
            "an elderly person sitting on a bed",
            "an elderly person getting out of bed safely",
        ]
    if num_frames is None:
        num_frames = 8
    if temperature is None:
        temperature = 1.0
    if aggregation is None:
        aggregation = "mean"
    
    logger.info(f"Processing video: {video_path}")
    logger.info(f"Using {len(prompts)} prompts with {num_frames} frames")
    
    # Load and preprocess video frames
    try:
        video_frames = get_clip_inputs_from_video(
            video_path=video_path,
            num_frames=num_frames,
            processor=vlm.processor
        )
        logger.info(f"Loaded {video_frames.shape[0]} frames")
    except Exception as e:
        logger.error(f"Failed to load video: {e}")
        raise
    
    # Get probabilities for each prompt
    probabilities = vlm.get_text_probabilities_for_video(
        video_frames=video_frames,
        prompts=prompts,
        aggregation=aggregation,
        temperature=temperature
    )
    
    # Identify fall-related prompts and compute aggregate fall probability
    # This is a heuristic: look for prompts containing "fall" or "falling"
    fall_prompts = [
        prompt for prompt in prompts
        if "fall" in prompt.lower() or "falling" in prompt.lower()
    ]
    
    if fall_prompts:
        # Sum probabilities of all fall-related prompts
        fall_probability = sum(probabilities.get(prompt, 0.0) for prompt in fall_prompts)
        probabilities["fall_probability"] = fall_probability
    else:
        # If no explicit fall prompts, use the first prompt as fall indicator
        # (assuming prompts are ordered with fall first)
        probabilities["fall_probability"] = probabilities.get(prompts[0], 0.0)
    
    return probabilities


def zero_shot_activity_recognition(
    video_path: str,
    vlm: VisionLanguageModel,
    activity_prompts: Dict[str, List[str]],
    num_frames: Optional[int] = None,
    temperature: Optional[float] = None,
    aggregation: Optional[str] = None,
    config: Optional[Dict] = None
) -> Dict[str, Dict[str, float]]:
    """
    Generic zero-shot activity recognition for multiple activity categories.
    
    This is a more general version that can handle multiple activity categories
    (e.g., "fall", "walking", "sitting", etc.) with multiple prompts per category.
    
    Args:
        video_path: Path to video file.
        vlm: Initialized VisionLanguageModel instance.
        activity_prompts: Dictionary mapping activity names to lists of prompts.
                         Example: {"fall": ["person falling", ...], "walking": [...]}
        num_frames: Number of frames to sample.
        temperature: Temperature for softmax.
        aggregation: Aggregation method.
        config: Configuration dictionary.
    
    Returns:
        Dictionary mapping activity names to their probabilities.
        Each activity probability is the average probability of its prompts.
    """
    # Flatten prompts for CLIP
    all_prompts = []
    prompt_to_activity = {}
    
    for activity, prompts in activity_prompts.items():
        for prompt in prompts:
            all_prompts.append(prompt)
            prompt_to_activity[prompt] = activity
    
    # Get probabilities for all prompts
    prompt_probs = zero_shot_fall_detection(
        video_path=video_path,
        vlm=vlm,
        prompts=all_prompts,
        num_frames=num_frames,
        temperature=temperature,
        aggregation=aggregation,
        config=config
    )
    
    # Aggregate by activity
    activity_probs = {}
    for activity in activity_prompts.keys():
        activity_prompts_list = activity_prompts[activity]
        # Average probability across all prompts for this activity
        activity_prob = sum(
            prompt_probs.get(prompt, 0.0)
            for prompt in activity_prompts_list
        ) / len(activity_prompts_list)
        activity_probs[activity] = activity_prob
    
    return activity_probs


def get_fall_detection_result(
    probabilities: Dict[str, float],
    threshold: float = 0.5
) -> Dict[str, any]:
    """
    Convert probabilities to a fall detection result.
    
    Args:
        probabilities: Dictionary from zero_shot_fall_detection.
        threshold: Threshold for binary classification (default: 0.5).
    
    Returns:
        Dictionary with:
        - "is_fall": Boolean indicating if fall was detected.
        - "fall_probability": Probability of fall.
        - "all_probabilities": Original probabilities dict.
    """
    fall_prob = probabilities.get("fall_probability", 0.0)
    is_fall = fall_prob >= threshold
    
    return {
        "is_fall": is_fall,
        "fall_probability": fall_prob,
        "all_probabilities": probabilities,
    }











