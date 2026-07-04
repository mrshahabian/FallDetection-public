"""
Train few-shot VLM classifier on fall videos.
Randomly selects 10 shots for training.
"""

import os
import sys
import random
import argparse
import logging
from pathlib import Path
from typing import List, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from vlm.vlm_model import VisionLanguageModel
from vlm.few_shot_train import train_few_shot_classifier, save_classifier
from vlm.config import load_config, get_device, get_clip_model_name, get_num_frames
from evaluation.video_sampler import get_video_files

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def prepare_few_shot_dataset(
    fall_base_dir: str,
    non_fall_videos_dir: str,
    num_shots: int = 10,
    seed: int = 42
) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
    """
    Prepare few-shot dataset by randomly sampling videos.
    
    Args:
        fall_videos_dir: Directory containing fall videos
        non_fall_videos_dir: Directory containing non-fall videos
        num_shots: Number of shots per class (default: 10)
        seed: Random seed
    
    Returns:
        Tuple of (train_videos, val_videos) where each is a list of (video_path, label) tuples
        label: 0 for non-fall, 1 for fall
    """
    random.seed(seed)
    
    # Get all fall videos from multiple directories
    # For VLM training, use only single-person dataset (Raw_Video)
    from evaluation.dataset_sampler import get_all_fall_videos
    fall_videos = get_all_fall_videos(fall_base_dir, single_person_only=True)
    logger.info(f"Using single-person dataset only for VLM training (Raw_Video)")
    non_fall_videos = get_video_files(non_fall_videos_dir) if non_fall_videos_dir and os.path.exists(non_fall_videos_dir) else []
    
    logger.info(f"Found {len(fall_videos)} fall videos and {len(non_fall_videos)} non-fall videos")
    
    # Sample videos
    if len(fall_videos) < num_shots:
        logger.warning(f"Only {len(fall_videos)} fall videos available, using all")
        sampled_fall = fall_videos
    else:
        sampled_fall = random.sample(fall_videos, num_shots)
    
    if len(non_fall_videos) < num_shots:
        logger.warning(f"Only {len(non_fall_videos)} non-fall videos available, using all")
        sampled_non_fall = non_fall_videos
    else:
        sampled_non_fall = random.sample(non_fall_videos, num_shots)
    
    # Create labeled pairs: (video_path, label)
    # Use 80% for training, 20% for validation
    train_fall = sampled_fall[:int(len(sampled_fall) * 0.8)]
    val_fall = sampled_fall[int(len(sampled_fall) * 0.8):]
    
    train_non_fall = sampled_non_fall[:int(len(sampled_non_fall) * 0.8)]
    val_non_fall = sampled_non_fall[int(len(sampled_non_fall) * 0.8):]
    
    train_videos = [(v, 1) for v in train_fall] + [(v, 0) for v in train_non_fall]
    val_videos = [(v, 1) for v in val_fall] + [(v, 0) for v in val_non_fall]
    
    # Shuffle
    random.shuffle(train_videos)
    random.shuffle(val_videos)
    
    logger.info(f"Training set: {len(train_videos)} videos ({len(train_fall)} fall, {len(train_non_fall)} non-fall)")
    logger.info(f"Validation set: {len(val_videos)} videos ({len(val_fall)} fall, {len(val_non_fall)} non-fall)")
    
    return train_videos, val_videos


def main():
    parser = argparse.ArgumentParser(description="Train few-shot VLM classifier")
    parser.add_argument("--fall_base_dir", type=str,
                       default="/home/reza/Documents/Datasets/Fall/Fall",
                       help="Base directory containing fall videos (may have Raw_Video and Raw_Video_part2)")
    parser.add_argument("--non_fall_videos_dir", type=str,
                       default="/home/reza/Documents/Datasets/Fall/No_Fall/Raw_Video",
                       help="Directory containing non-fall videos")
    parser.add_argument("--num_shots", type=int, default=10,
                       help="Number of shots per class (default: 10)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--config", type=str, default=None,
                       help="Path to VLM config file")
    parser.add_argument("--save_path", type=str,
                       default="evaluation/checkpoints/few_shot_vlm_classifier.pt",
                       help="Path to save trained classifier")
    parser.add_argument("--num_frames", type=int, default=1,
                       help="Number of frames to sample per video (default: 1)")
    
    args = parser.parse_args()
    
    # Load config
    if args.config:
        config = load_config(args.config)
    else:
        config_path = project_root / "configs" / "vlm_config.yaml"
        if config_path.exists():
            config = load_config(str(config_path))
        else:
            config = {}
    
    # Get device and model name
    device = get_device(config)
    model_name = get_clip_model_name(config)
    
    # Initialize VLM
    logger.info(f"Initializing VLM model: {model_name} on {device}")
    vlm = VisionLanguageModel(model_name=model_name, device=device)
    
    # Prepare dataset
    train_videos, val_videos = prepare_few_shot_dataset(
        args.fall_base_dir,
        args.non_fall_videos_dir,
        args.num_shots,
        args.seed
    )
    
    # Update config with num_frames
    if 'video' not in config:
        config['video'] = {}
    config['video']['num_frames'] = args.num_frames
    
    # Train classifier
    logger.info("Training few-shot classifier...")
    classifier = train_few_shot_classifier(
        train_videos=train_videos,
        vlm=vlm,
        val_videos=val_videos if len(val_videos) > 0 else None,
        config=config,
        hidden_dim=None,  # Linear classifier
        save_path=args.save_path
    )
    
    logger.info(f"Training completed! Classifier saved to {args.save_path}")


if __name__ == "__main__":
    main()

