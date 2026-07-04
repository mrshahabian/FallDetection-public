"""
Vision-Language Model (VLM) based fall detection module.

This module provides zero-shot and few-shot activity recognition capabilities
using CLIP (Contrastive Language-Image Pre-training) models from HuggingFace.
"""

__version__ = "0.1.0"

from .vlm_model import VisionLanguageModel
from .zero_shot_inference import zero_shot_fall_detection
from .few_shot_train import (
    extract_video_embedding,
    build_feature_dataset,
    train_few_shot_classifier,
    few_shot_predict,
    save_classifier,
    load_classifier,
)

__all__ = [
    "VisionLanguageModel",
    "zero_shot_fall_detection",
    "extract_video_embedding",
    "build_feature_dataset",
    "train_few_shot_classifier",
    "few_shot_predict",
    "save_classifier",
    "load_classifier",
]











