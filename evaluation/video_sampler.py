"""
Video sampler for randomly selecting test videos from fall dataset.
"""

import os
import random
from pathlib import Path
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


def get_video_files(directory: str, extensions: List[str] = None) -> List[str]:
    """
    Get all video files from a directory recursively.
    
    Args:
        directory: Root directory to search
        extensions: List of video file extensions (default: ['.mp4', '.avi', '.mov', '.mkv'])
    
    Returns:
        List of video file paths
    """
    if extensions is None:
        extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.m4v']
    
    video_files = []
    directory_path = Path(directory)
    
    if not directory_path.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    for ext in extensions:
        video_files.extend(directory_path.rglob(f"*{ext}"))
        video_files.extend(directory_path.rglob(f"*{ext.upper()}"))
    
    return [str(f) for f in video_files]


def sample_videos(
    video_dir: str,
    num_samples: int = 100,
    seed: Optional[int] = None,
    extensions: List[str] = None
) -> List[str]:
    """
    Randomly sample videos from a directory.
    
    Args:
        video_dir: Directory containing videos
        num_samples: Number of videos to sample (default: 100)
        seed: Random seed for reproducibility
        extensions: List of video file extensions
    
    Returns:
        List of sampled video file paths
    """
    if seed is not None:
        random.seed(seed)
    
    # Get all video files
    all_videos = get_video_files(video_dir, extensions)
    
    if len(all_videos) == 0:
        raise ValueError(f"No video files found in {video_dir}")
    
    logger.info(f"Found {len(all_videos)} video files in {video_dir}")
    
    # Sample videos
    if num_samples > len(all_videos):
        logger.warning(
            f"Requested {num_samples} samples but only {len(all_videos)} videos available. "
            f"Using all {len(all_videos)} videos."
        )
        sampled = all_videos
    else:
        sampled = random.sample(all_videos, num_samples)
    
    logger.info(f"Sampled {len(sampled)} videos for evaluation")
    
    return sampled


def save_sampled_videos(video_paths: List[str], save_path: str):
    """
    Save list of sampled video paths to a file.
    
    Args:
        video_paths: List of video file paths
        save_path: Path to save the list
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        for path in video_paths:
            f.write(f"{path}\n")
    logger.info(f"Saved {len(video_paths)} video paths to {save_path}")


def load_sampled_videos(load_path: str) -> List[str]:
    """
    Load list of sampled video paths from a file.
    
    Args:
        load_path: Path to load the list from
    
    Returns:
        List of video file paths
    """
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"File not found: {load_path}")
    
    with open(load_path, 'r') as f:
        video_paths = [line.strip() for line in f if line.strip()]
    
    logger.info(f"Loaded {len(video_paths)} video paths from {load_path}")
    return video_paths


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Sample videos from fall dataset")
    parser.add_argument("--video_dir", type=str, 
                       default="/home/reza/Documents/Datasets/Fall/Fall/Raw_Video",
                       help="Directory containing fall videos")
    parser.add_argument("--num_samples", type=int, default=100,
                       help="Number of videos to sample")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--save_path", type=str,
                       default="evaluation/data/sampled_videos.txt",
                       help="Path to save sampled video list")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    videos = sample_videos(args.video_dir, args.num_samples, args.seed)
    save_sampled_videos(videos, args.save_path)
    
    print(f"\nSampled {len(videos)} videos:")
    for i, video in enumerate(videos[:5], 1):
        print(f"  {i}. {os.path.basename(video)}")
    if len(videos) > 5:
        print(f"  ... and {len(videos) - 5} more")



