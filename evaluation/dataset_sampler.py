"""
Dataset sampler for balanced sampling from fall and non-fall videos.
Handles multiple fall video directories and creates train/test splits.
"""

import os
import random
from pathlib import Path
from typing import List, Tuple, Optional
import logging

from .video_sampler import get_video_files, save_sampled_videos

logger = logging.getLogger(__name__)

# KTH dataset activities (all are non-fall activities)
KTH_ACTIVITIES = ['boxing', 'handclapping', 'handwaving', 'jogging', 'running', 'walking']


def get_kth_videos(kth_base_dir: str, activities: List[str] = None) -> List[str]:
    """
    Get all videos from KTH dataset.
    
    KTH dataset structure:
    /path/to/KTH/
    ├── boxing/
    ├── handclapping/
    ├── handwaving/
    ├── jogging/
    ├── running/
    └── walking/
    
    Args:
        kth_base_dir: Base directory of KTH dataset
        activities: List of activities to include (default: all 6 activities)
    
    Returns:
        List of all KTH video file paths
    """
    if activities is None:
        activities = KTH_ACTIVITIES
    
    kth_videos = []
    
    if not os.path.exists(kth_base_dir):
        logger.warning(f"KTH dataset directory not found: {kth_base_dir}")
        return kth_videos
    
    for activity in activities:
        activity_dir = os.path.join(kth_base_dir, activity)
        if os.path.exists(activity_dir):
            videos = get_video_files(activity_dir)
            kth_videos.extend(videos)
            logger.info(f"Found {len(videos)} videos in {activity_dir} ({activity})")
        else:
            logger.warning(f"Activity directory not found: {activity_dir}")
    
    logger.info(f"Total KTH videos collected: {len(kth_videos)}")
    return kth_videos


def get_all_fall_videos(fall_base_dir: str, single_person_only: bool = True) -> List[str]:
    """
    Get all fall videos from multiple directories.
    
    Args:
        fall_base_dir: Base directory containing fall videos
                      (may have Raw_Video and Raw_Video_part2 subdirectories)
        single_person_only: If True, only return videos from Raw_Video (single-person dataset)
    
    Returns:
        List of all fall video file paths
    """
    fall_videos = []
    
    # Check for Raw_Video directory (single-person dataset)
    raw_video_dir = os.path.join(fall_base_dir, "Raw_Video")
    if os.path.exists(raw_video_dir):
        videos = get_video_files(raw_video_dir)
        fall_videos.extend(videos)
        logger.info(f"Found {len(videos)} videos in {raw_video_dir}")
    
    if not single_person_only:
        # Check for Raw_Video_part2 directory (multi-person dataset)
        raw_video_part2_dir = os.path.join(fall_base_dir, "Raw_Video_part2")
        if os.path.exists(raw_video_part2_dir):
            videos = get_video_files(raw_video_part2_dir)
            fall_videos.extend(videos)
            logger.info(f"Found {len(videos)} videos in {raw_video_part2_dir}")
        
        # Also check if the base directory itself contains videos
        videos = get_video_files(fall_base_dir)
        if videos:
            fall_videos.extend(videos)
            logger.info(f"Found {len(videos)} videos in {fall_base_dir}")
    
    return fall_videos


def sample_balanced_dataset(
    fall_base_dir: str,
    non_fall_dir: str = None,
    kth_dir: str = None,
    num_fall: int = 100,
    num_non_fall: int = 100,
    seed: Optional[int] = None,
    single_person_only: bool = True
) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
    """
    Sample balanced dataset with fall and non-fall videos.
    Supports both regular non-fall directory and KTH dataset.
    
    Args:
        fall_base_dir: Base directory containing fall videos
        non_fall_dir: Directory containing non-fall videos (optional)
        kth_dir: Base directory of KTH dataset (optional, if provided, uses KTH as non-fall)
        num_fall: Number of fall videos to sample
        num_non_fall: Number of non-fall videos to sample
        seed: Random seed for reproducibility
        single_person_only: If True, only use videos from Raw_Video (single-person dataset)
    
    Returns:
        Tuple of (train_videos, test_videos) where each is a list of (video_path, label) tuples
        label: 0 for non-fall, 1 for fall
    """
    if seed is not None:
        random.seed(seed)
    
    # Get all videos
    logger.info("Collecting fall videos...")
    all_fall_videos = get_all_fall_videos(fall_base_dir, single_person_only=single_person_only)
    
    # Collect non-fall videos
    all_non_fall_videos = []
    
    # Priority: KTH dataset if provided, otherwise use non_fall_dir
    if kth_dir and os.path.exists(kth_dir):
        logger.info("Collecting non-fall videos from KTH dataset...")
        all_non_fall_videos = get_kth_videos(kth_dir)
        logger.info(f"Using KTH dataset as non-fall samples: {len(all_non_fall_videos)} videos")
    elif non_fall_dir and os.path.exists(non_fall_dir):
        logger.info("Collecting non-fall videos...")
        all_non_fall_videos = get_video_files(non_fall_dir)
    else:
        logger.warning("No non-fall directory or KTH directory provided or found!")
    
    logger.info(f"Total fall videos: {len(all_fall_videos)}")
    logger.info(f"Total non-fall videos: {len(all_non_fall_videos)}")
    
    # Sample videos
    if len(all_fall_videos) < num_fall:
        logger.warning(f"Only {len(all_fall_videos)} fall videos available, using all")
        sampled_fall = all_fall_videos
    else:
        sampled_fall = random.sample(all_fall_videos, num_fall)
    
    if len(all_non_fall_videos) < num_non_fall:
        logger.warning(f"Only {len(all_non_fall_videos)} non-fall videos available, using all")
        sampled_non_fall = all_non_fall_videos
    else:
        sampled_non_fall = random.sample(all_non_fall_videos, num_non_fall)
    
    # Create labeled pairs: (video_path, label)
    # label: 0 = non-fall, 1 = fall
    fall_pairs = [(v, 1) for v in sampled_fall]
    non_fall_pairs = [(v, 0) for v in sampled_non_fall]
    
    # Combine and shuffle
    all_pairs = fall_pairs + non_fall_pairs
    random.shuffle(all_pairs)
    
    # Split into train/test (80/20)
    split_idx = int(len(all_pairs) * 0.8)
    train_pairs = all_pairs[:split_idx]
    test_pairs = all_pairs[split_idx:]
    
    logger.info(f"Train set: {len(train_pairs)} videos ({sum(1 for _, l in train_pairs if l == 1)} fall, {sum(1 for _, l in train_pairs if l == 0)} non-fall)")
    logger.info(f"Test set: {len(test_pairs)} videos ({sum(1 for _, l in test_pairs if l == 1)} fall, {sum(1 for _, l in test_pairs if l == 0)} non-fall)")
    
    return train_pairs, test_pairs


def save_dataset_split(
    train_pairs: List[Tuple[str, int]],
    test_pairs: List[Tuple[str, int]],
    save_dir: str
):
    """
    Save train/test split to files.
    
    Args:
        train_pairs: List of (video_path, label) tuples for training
        test_pairs: List of (video_path, label) tuples for testing
        save_dir: Directory to save the files
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Save train set
    train_path = os.path.join(save_dir, "train_videos.txt")
    with open(train_path, 'w') as f:
        for video_path, label in train_pairs:
            f.write(f"{video_path}\t{label}\n")
    logger.info(f"Saved {len(train_pairs)} training videos to {train_path}")
    
    # Save test set
    test_path = os.path.join(save_dir, "test_videos.txt")
    with open(test_path, 'w') as f:
        for video_path, label in test_pairs:
            f.write(f"{video_path}\t{label}\n")
    logger.info(f"Saved {len(test_pairs)} test videos to {test_path}")
    
    # Save test set for evaluation (video paths only, labels separate)
    test_videos_path = os.path.join(save_dir, "test_videos_list.txt")
    test_labels_path = os.path.join(save_dir, "test_labels.txt")
    
    with open(test_videos_path, 'w') as f:
        for video_path, _ in test_pairs:
            f.write(f"{video_path}\n")
    
    with open(test_labels_path, 'w') as f:
        for _, label in test_pairs:
            f.write(f"{label}\n")
    
    logger.info(f"Saved test video list to {test_videos_path}")
    logger.info(f"Saved test labels to {test_labels_path}")


def load_dataset_split(load_dir: str) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
    """
    Load train/test split from files.
    
    Args:
        load_dir: Directory containing the split files
    
    Returns:
        Tuple of (train_pairs, test_pairs)
    """
    train_path = os.path.join(load_dir, "train_videos.txt")
    test_path = os.path.join(load_dir, "test_videos.txt")
    
    train_pairs = []
    if os.path.exists(train_path):
        with open(train_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split('\t')
                    if len(parts) == 2:
                        train_pairs.append((parts[0], int(parts[1])))
    
    test_pairs = []
    if os.path.exists(test_path):
        with open(test_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split('\t')
                    if len(parts) == 2:
                        test_pairs.append((parts[0], int(parts[1])))
    
    logger.info(f"Loaded {len(train_pairs)} training videos and {len(test_pairs)} test videos")
    return train_pairs, test_pairs


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Sample balanced dataset from fall and non-fall videos")
    parser.add_argument("--fall_base_dir", type=str,
                       default="/home/reza/Documents/Datasets/Fall/Fall",
                       help="Base directory containing fall videos")
    parser.add_argument("--non_fall_dir", type=str,
                       default="/home/reza/Documents/Datasets/Fall/No_Fall/Raw_Video",
                       help="Directory containing non-fall videos")
    parser.add_argument("--num_fall", type=int, default=100,
                       help="Number of fall videos to sample")
    parser.add_argument("--num_non_fall", type=int, default=100,
                       help="Number of non-fall videos to sample")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--save_dir", type=str,
                       default="evaluation/data",
                       help="Directory to save dataset split")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    train_pairs, test_pairs = sample_balanced_dataset(
        args.fall_base_dir,
        args.non_fall_dir,
        args.num_fall,
        args.num_non_fall,
        args.seed
    )
    
    save_dataset_split(train_pairs, test_pairs, args.save_dir)
    
    print(f"\nDataset split created:")
    print(f"  Train: {len(train_pairs)} videos")
    print(f"  Test: {len(test_pairs)} videos")

