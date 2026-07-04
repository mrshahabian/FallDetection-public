"""
Create a shared test set that all models will use for fair comparison.
"""

import os
import sys
import argparse
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from evaluation.dataset_sampler import sample_balanced_dataset


def main():
    parser = argparse.ArgumentParser(description="Create shared test set for all models")
    parser.add_argument("--fall_base_dir", type=str,
                       default="/home/reza/Documents/Datasets/Fall/Fall",
                       help="Base directory containing fall videos")
    parser.add_argument("--kth_dir", type=str,
                       default="/home/reza/Documents/Datasets/KTH",
                       help="KTH dataset directory")
    parser.add_argument("--num_samples", type=int, default=100,
                       help="Number of samples per class (used for both if --num_fall/--num_non_fall not set)")
    parser.add_argument("--num_fall", type=int, default=None,
                       help="Number of fall videos to sample (overrides --num_samples for fall)")
    parser.add_argument("--num_non_fall", type=int, default=None,
                       help="Number of non-fall videos to sample (overrides --num_samples for non-fall)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--output_file", type=str,
                       default="evaluation/data/shared_test_set.txt",
                       help="Output file path")
    
    args = parser.parse_args()
    
    num_fall = args.num_fall if args.num_fall is not None else args.num_samples
    num_non_fall = args.num_non_fall if args.num_non_fall is not None else args.num_samples
    
    # Create shared test set (80/20 train/test split -> test has 20% of each)
    print(f"Creating shared test set: {num_fall} fall + {num_non_fall} non-fall (80/20 split -> 40 test videos for 100+100)")
    print(f"Using seed: {args.seed}")
    
    _, test_pairs = sample_balanced_dataset(
        args.fall_base_dir,
        non_fall_dir=None,
        kth_dir=args.kth_dir,
        num_fall=num_fall,
        num_non_fall=num_non_fall,
        seed=args.seed,
        single_person_only=True  # Single-person dataset only
    )
    
    # Save test set
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w') as f:
        for video_path, label in test_pairs:
            f.write(f"{video_path}\t{label}\n")
    
    print(f"✓ Saved {len(test_pairs)} test videos to {args.output_file}")
    print(f"  - Fall videos: {sum(1 for _, l in test_pairs if l == 1)}")
    print(f"  - Non-fall videos: {sum(1 for _, l in test_pairs if l == 0)}")


if __name__ == "__main__":
    main()


