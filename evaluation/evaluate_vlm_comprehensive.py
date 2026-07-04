"""
Comprehensive VLM evaluation (single-person dataset only - best fitted process):
- Single-person dataset only (Raw_Video + KTH)
- For each method: 1 and 8 frames per window
- Total: 2 methods × 1 condition × 2 frame counts = 4 evaluations
"""

import os
import sys
import argparse
import logging
import subprocess
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from evaluation.evaluate_all import evaluate_method, generate_comparison_table
from evaluation.evaluators import ZeroShotVLMEvaluator, FewShotVLMEvaluator
from evaluation.dataset_sampler import sample_balanced_dataset, get_all_fall_videos
from evaluation.video_sampler import get_video_files
from evaluation.metrics import calculate_video_level_metrics

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def prepare_videos_condition_a(fall_base_dir: str, non_fall_dir: str = None, kth_dir: str = None, num_samples: int = 100, seed: int = 42):
    """
    Condition A: Only Part1 (single-person dataset) fall videos + all non-fall videos.
    Supports KTH dataset as non-fall samples.
    
    Returns:
        Tuple of (video_paths, true_labels)
    """
    import random
    from evaluation.dataset_sampler import get_kth_videos
    random.seed(seed)
    
    # Get only Part1 fall videos
    part1_dir = os.path.join(fall_base_dir, "Raw_Video")
    part1_fall_videos = get_video_files(part1_dir)
    
    # Get non-fall videos (KTH has priority if provided)
    non_fall_videos = []
    if kth_dir and os.path.exists(kth_dir):
        logger.info(f"Condition A: Using KTH dataset as non-fall samples")
        non_fall_videos = get_kth_videos(kth_dir)
    elif non_fall_dir and os.path.exists(non_fall_dir):
        non_fall_videos = get_video_files(non_fall_dir)
    
    logger.info(f"Condition A: Found {len(part1_fall_videos)} Part1 fall videos, {len(non_fall_videos)} non-fall videos")
    
    # Sample
    num_fall = min(num_samples, len(part1_fall_videos))
    num_non_fall = min(num_samples, len(non_fall_videos))
    
    sampled_fall = random.sample(part1_fall_videos, num_fall) if len(part1_fall_videos) >= num_fall else part1_fall_videos
    sampled_non_fall = random.sample(non_fall_videos, num_non_fall) if len(non_fall_videos) >= num_non_fall else non_fall_videos
    
    # Create labeled pairs
    video_paths = sampled_fall + sampled_non_fall
    true_labels = [True] * len(sampled_fall) + [False] * len(sampled_non_fall)
    
    # Shuffle
    combined = list(zip(video_paths, true_labels))
    random.shuffle(combined)
    video_paths, true_labels = zip(*combined)
    
    logger.info(f"Condition A: Selected {len(sampled_fall)} fall + {len(sampled_non_fall)} non-fall = {len(video_paths)} total videos")
    
    return list(video_paths), list(true_labels)


# Condition B removed - using single-person dataset only (best fitted process)


def evaluate_vlm_method(
    method_name: str,
    evaluator,
    video_paths: List[str],
    true_labels: List[bool],
    condition: str,
    num_frames: int
) -> Dict[str, Any]:
    """
    Evaluate a VLM method with given parameters.
    
    For inference time calculation: Always uses 1 frame per window (for fair comparison)
    For accuracy evaluation: Uses num_frames (e.g., 8 frames)
    """
    from evaluation.evaluate_all import evaluate_method
    
    logger.info(f"\n{'='*80}")
    logger.info(f"{method_name} - Condition {condition} - {num_frames} frame(s) per window")
    logger.info(f"{'='*80}")
    logger.info(f"  Videos: {len(video_paths)} ({sum(true_labels)} fall, {len(true_labels) - sum(true_labels)} non-fall)")
    logger.info(f"  Accuracy evaluation: {num_frames} frame(s)")
    logger.info(f"  Inference time calculation: 1 frame (for fair comparison)")
    
    # Evaluate for accuracy with num_frames
    result = evaluate_method(
        evaluator,
        video_paths,
        f"{method_name} ({condition}, {num_frames} frame(s))",
        true_labels
    )
    
    # Calculate inference time separately with 1 frame (for fair comparison)
    if num_frames != 1:
        from evaluation.evaluators import ZeroShotVLMEvaluator, FewShotVLMEvaluator
        logger.info(f"  Calculating inference time with 1 frame for fair comparison...")
        
        # Create evaluator with 1 frame for inference time
        if isinstance(evaluator, ZeroShotVLMEvaluator):
            inference_evaluator = ZeroShotVLMEvaluator(
                window_length=evaluator.window_length,
                overlap=evaluator.overlap,
                num_frames=1,  # Always 1 frame for inference time
                device=str(evaluator.vlm.device)
            )
        elif isinstance(evaluator, FewShotVLMEvaluator):
            inference_evaluator = FewShotVLMEvaluator(
                classifier_path=evaluator.classifier_path,
                window_length=evaluator.window_length,
                overlap=evaluator.overlap,
                num_frames=1,  # Always 1 frame for inference time
                device=str(evaluator.vlm.device)
            )
        else:
            inference_evaluator = evaluator
        
        # Get inference times with 1 frame
        inference_results = inference_evaluator.evaluate_batch(video_paths)
        inference_times = []
        for video_result in inference_results.get('per_video_results', []):
            inference_times.extend(video_result.get('inference_times', []))
        
        if inference_times:
            # Update metrics with inference time from 1-frame evaluation
            result['metrics']['avg_inference_time'] = float(np.mean(inference_times))
            result['metrics']['total_inference_time'] = float(np.sum(inference_times))
            result['metrics']['num_windows'] = len(inference_times)
            logger.info(f"  Inference time (1 frame): {result['metrics']['avg_inference_time']:.4f}s per window")
    
    result['condition'] = condition
    result['num_frames'] = num_frames
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Comprehensive VLM evaluation with multiple conditions")
    parser.add_argument("--fall_base_dir", type=str,
                       default="/home/reza/Documents/Datasets/Fall/Fall",
                       help="Base directory containing fall videos")
    parser.add_argument("--non_fall_dir", type=str, default=None,
                       help="Directory containing non-fall videos (optional if --kth_dir is provided)")
    parser.add_argument("--kth_dir", type=str, default=None,
                       help="Base directory of KTH dataset (will be used as non-fall samples if provided)")
    parser.add_argument("--num_samples", type=int, default=100,
                       help="Number of videos to sample per class")
    parser.add_argument("--shared_test_set", type=str, default="evaluation/data/shared_test_set.txt",
                       help="Path to shared test set file (ensures all models use same samples for fair comparison)")
    parser.add_argument("--few_shot_classifier", type=str,
                       default="evaluation/checkpoints/few_shot_vlm_classifier.pt",
                       help="Path to few-shot VLM classifier")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--window_length", type=int, default=32,
                       help="Window length in frames")
    parser.add_argument("--overlap", type=float, default=0.5,
                       help="Overlap ratio between windows")
    parser.add_argument("--num_frames", type=int, default=8,
                       help="Number of frames per window for VLM (default: 8)")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use for inference")
    parser.add_argument("--results_dir", type=str,
                       default="evaluation/results/vlm_comprehensive",
                       help="Directory to save results")
    
    args = parser.parse_args()
    
    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Frame counts to test (use single value if specified, otherwise test all)
    if args.num_frames and args.num_frames > 0:
        frame_counts = [args.num_frames]
    else:
        frame_counts = [1, 2, 4, 8]
    
    # Single condition: Single-person dataset only (best fitted process)
    conditions = {
        'A': ('Single-Person Only', prepare_videos_condition_a)
    }
    
    all_results = []
    
    # Prepare videos for each condition
    # Use shared test set if available (ensures all models use same samples)
    condition_videos = {}
    for cond_id, (cond_name, prep_func) in conditions.items():
        logger.info(f"\n{'='*80}")
        logger.info(f"Preparing videos for Condition {cond_id}: {cond_name}")
        logger.info(f"{'='*80}")
        
        if args.shared_test_set and os.path.exists(args.shared_test_set):
            # Use shared test set (ensures all models use same samples)
            logger.info(f"Using shared test set from {args.shared_test_set}")
            video_paths = []
            true_labels = []
            with open(args.shared_test_set, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split('\t')
                        if len(parts) == 2:
                            video_paths.append(parts[0])
                            true_labels.append(bool(int(parts[1])))
            logger.info(f"Loaded {len(video_paths)} videos from shared test set")
            logger.info(f"  Fall videos: {sum(true_labels)}, Non-fall videos: {len(true_labels) - sum(true_labels)}")
        else:
            # Sample new test set
            video_paths, true_labels = prep_func(
                args.fall_base_dir,
                non_fall_dir=args.non_fall_dir,
                kth_dir=args.kth_dir,
                num_samples=args.num_samples,
                seed=args.seed
            )
        
        condition_videos[cond_id] = (video_paths, true_labels, cond_name)
        
        # Save test videos for this condition
        test_list_path = os.path.join(args.results_dir, f"test_videos_condition_{cond_id}.txt")
        with open(test_list_path, 'w') as f:
            for video_path, label in zip(video_paths, true_labels):
                f.write(f"{video_path}\t{int(label)}\n")
        logger.info(f"Saved test videos for Condition {cond_id} to {test_list_path}")
    
    # Also save combined test videos (use Condition A as primary)
    if 'A' in condition_videos:
        video_paths_a, true_labels_a, _ = condition_videos['A']
        test_list_path = os.path.join(args.results_dir, "test_videos.txt")
        with open(test_list_path, 'w') as f:
            for video_path, label in zip(video_paths_a, true_labels_a):
                f.write(f"{video_path}\t{int(label)}\n")
        logger.info(f"Saved primary test videos to {test_list_path}")
    
    # ========================================================================
    # Zero-Shot VLM Evaluation
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("ZERO-SHOT VLM EVALUATION")
    logger.info("="*80)
    
    for cond_id, (video_paths, true_labels, cond_name) in condition_videos.items():
        for num_frames in frame_counts:
            try:
                evaluator = ZeroShotVLMEvaluator(
                    window_length=args.window_length,
                    overlap=args.overlap,
                    num_frames=num_frames,  # Use current frame count from loop
                    device=args.device
                )
                
                result = evaluate_vlm_method(
                    "Zero-Shot VLM",
                    evaluator,
                    video_paths,
                    true_labels,
                    f"{cond_id} ({cond_name})",
                    num_frames
                )
                
                all_results.append(result)
            except Exception as e:
                logger.error(f"Error evaluating Zero-Shot VLM (Condition {cond_id}, {num_frames} frames): {e}", exc_info=True)
    
    # ========================================================================
    # Few-Shot VLM Evaluation
    # ========================================================================
    if os.path.exists(args.few_shot_classifier):
        logger.info("\n" + "="*80)
        logger.info("FEW-SHOT VLM EVALUATION")
        logger.info("="*80)
        
        for cond_id, (video_paths, true_labels, cond_name) in condition_videos.items():
            for num_frames in frame_counts:
                try:
                    evaluator = FewShotVLMEvaluator(
                        classifier_path=args.few_shot_classifier,
                        window_length=args.window_length,
                        overlap=args.overlap,
                        num_frames=num_frames,  # Use current frame count from loop
                        device=args.device
                    )
                    
                    result = evaluate_vlm_method(
                        "Few-Shot VLM",
                        evaluator,
                        video_paths,
                        true_labels,
                        f"{cond_id} ({cond_name})",
                        num_frames
                    )
                    
                    all_results.append(result)
                except Exception as e:
                    logger.error(f"Error evaluating Few-Shot VLM (Condition {cond_id}, {num_frames} frames): {e}", exc_info=True)
    else:
        logger.warning(f"Few-shot classifier not found at {args.few_shot_classifier}. Skipping Few-Shot VLM evaluation.")
    
    # ========================================================================
    # Generate Results
    # ========================================================================
    if len(all_results) > 0:
        logger.info("\n" + "="*80)
        logger.info("GENERATING COMPARISON TABLES")
        logger.info("="*80)
        
        # Overall comparison table
        table_path = os.path.join(args.results_dir, "comparison_table_all.txt")
        generate_comparison_table(all_results, table_path)
        
        # Separate tables by condition and method
        for method_name in ["Zero-Shot VLM", "Few-Shot VLM"]:
            method_results = [r for r in all_results if method_name in r['method_name']]
            if method_results:
                # By condition
                for cond_id in ['A', 'B']:
                    cond_results = [r for r in method_results if f"Condition {cond_id}" in r['method_name']]
                    if cond_results:
                        cond_name = conditions[cond_id][0]
                        table_path = os.path.join(args.results_dir, f"{method_name.replace(' ', '_')}_Condition_{cond_id}_{cond_name.replace(' ', '_')}.txt")
                        generate_comparison_table(cond_results, table_path)
                
                # By frame count
                for num_frames in frame_counts:
                    frame_results = [r for r in method_results if f"{num_frames} frame" in r['method_name']]
                    if frame_results:
                        table_path = os.path.join(args.results_dir, f"{method_name.replace(' ', '_')}_{num_frames}frames.txt")
                        generate_comparison_table(frame_results, table_path)
        
        # Save detailed results as JSON
        json_path = os.path.join(args.results_dir, "detailed_results.json")
        json_results = []
        for result in all_results:
            json_result = {
                'method_name': result['method_name'],
                'condition': result.get('condition', ''),
                'num_frames': result.get('num_frames', 0),
                'num_videos': result['num_videos'],
                'metrics': result['metrics'],
                'total_evaluation_time': result['total_evaluation_time']
            }
            json_results.append(json_result)
        
        with open(json_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        logger.info(f"Detailed results saved to {json_path}")
        
        # Print summary
        logger.info("\n" + "="*80)
        logger.info("EVALUATION SUMMARY")
        logger.info("="*80)
        logger.info(f"Total evaluations: {len(all_results)}")
        logger.info(f"  - Zero-Shot VLM: {len([r for r in all_results if 'Zero-Shot' in r['method_name']])}")
        logger.info(f"  - Few-Shot VLM: {len([r for r in all_results if 'Few-Shot' in r['method_name']])}")
        logger.info(f"\nResults saved to: {args.results_dir}")
        logger.info("="*80)
    else:
        logger.error("No results to save!")


if __name__ == "__main__":
    main()


