#!/usr/bin/env python3
"""
Test script to run fall detection on all videos in a directory and generate a summary report.
"""

import sys
import os
from pathlib import Path
import argparse

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from vlm.vlm_model import VisionLanguageModel
from vlm.zero_shot_inference import zero_shot_fall_detection
from vlm.config import load_config, get_device, get_clip_model_name, get_few_shot_config
import logging

logging.basicConfig(level=logging.WARNING)  # Reduce verbosity for batch testing


def test_videos_in_directory(directory: str, threshold: float = 0.45):
    """
    Test all videos in a directory and return results.
    
    Args:
        directory: Path to directory containing videos
        threshold: Classification threshold
        
    Returns:
        List of (video_name, fall_prob, is_fall) tuples
    """
    video_files = sorted([f for f in os.listdir(directory) if f.endswith(('.mp4', '.avi', '.mov'))])
    
    if not video_files:
        print(f"No video files found in {directory}")
        return []
    
    # Load config and initialize VLM
    config = load_config()
    device = get_device(config)
    model_name = get_clip_model_name(config)
    
    print(f"Initializing CLIP model: {model_name}")
    print(f"Using device: {device}\n")
    
    vlm = VisionLanguageModel(model_name=model_name, device=device)
    
    results = []
    
    for video_file in video_files:
        video_path = os.path.join(directory, video_file)
        try:
            probabilities = zero_shot_fall_detection(
                video_path=video_path,
                vlm=vlm,
                config=config
            )
            
            fall_prob = probabilities.get("fall_probability", 0.0)
            is_fall = fall_prob >= threshold
            
            results.append((video_file, fall_prob, is_fall))
            
            status = "✓ FALL" if is_fall else "✗ NO FALL"
            print(f"{video_file:40s} | Prob: {fall_prob:.3f} | {status}")
            
        except Exception as e:
            print(f"{video_file:40s} | ERROR: {e}")
            results.append((video_file, None, None))
    
    return results


def print_summary(results):
    """Print summary statistics."""
    valid_results = [r for r in results if r[1] is not None]
    
    if not valid_results:
        print("\nNo valid results to summarize.")
        return
    
    fall_detected = sum(1 for _, _, is_fall in valid_results if is_fall)
    total = len(valid_results)
    
    probs = [prob for _, prob, _ in valid_results if prob is not None]
    avg_prob = sum(probs) / len(probs) if probs else 0
    min_prob = min(probs) if probs else 0
    max_prob = max(probs) if probs else 0
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total videos tested: {total}")
    print(f"Falls detected: {fall_detected} ({100*fall_detected/total:.1f}%)")
    print(f"Average fall probability: {avg_prob:.3f}")
    print(f"Min fall probability: {min_prob:.3f}")
    print(f"Max fall probability: {max_prob:.3f}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Test fall detection on all videos in a directory")
    parser.add_argument("directory", type=str, help="Directory containing videos")
    parser.add_argument("--threshold", type=float, default=0.45, help="Classification threshold")
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.directory):
        print(f"Error: Directory not found: {args.directory}")
        sys.exit(1)
    
    print(f"Testing videos in: {args.directory}")
    print(f"Threshold: {args.threshold}\n")
    
    results = test_videos_in_directory(args.directory, args.threshold)
    print_summary(results)


if __name__ == "__main__":
    main()











