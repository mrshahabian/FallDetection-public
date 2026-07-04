"""
Main evaluation script that runs all fall detection methods and generates comparison tables.
"""

import os
import sys
import argparse
import logging
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from evaluation.video_sampler import load_sampled_videos, sample_videos
from evaluation.evaluators import (
    SimpleFallEvaluator,
    ZeroShotVLMEvaluator,
    FewShotVLMEvaluator,
    CNNEvaluator
)
from evaluation.metrics import calculate_video_level_metrics

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def evaluate_method(
    evaluator,
    video_paths: List[str],
    method_name: str,
    true_labels: List[bool] = None
) -> Dict[str, Any]:
    """
    Evaluate a single method on all videos.
    
    Args:
        evaluator: Evaluator instance
        video_paths: List of video paths
        method_name: Name of the method
        true_labels: Optional true labels (if None, assumes all are falls)
    
    Returns:
        Dictionary with evaluation results
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluating {method_name}")
    logger.info(f"{'='*60}")
    
    start_time = time.time()
    
    # Evaluate batch
    batch_results = evaluator.evaluate_batch(video_paths)
    
    total_time = time.time() - start_time
    
    # Calculate metrics
    per_video_results = batch_results.get('per_video_results', [])
    
    if true_labels is None:
        # Assume all videos are falls
        true_labels = [True] * len(per_video_results)
    
    metrics = calculate_video_level_metrics(per_video_results, true_labels)
    
    # Add summary statistics
    results = {
        'method_name': method_name,
        'num_videos': len(per_video_results),
        'metrics': metrics,
        'batch_results': batch_results,
        'total_evaluation_time': total_time,
        'per_video_results': per_video_results
    }
    
    logger.info(f"\n{method_name} Results:")
    logger.info(f"  Accuracy: {metrics.get('accuracy', 0.0):.4f}")
    logger.info(f"  Precision: {metrics.get('precision', 0.0):.4f}")
    logger.info(f"  Recall: {metrics.get('recall', 0.0):.4f}")
    logger.info(f"  F1-Score: {metrics.get('f1_score', 0.0):.4f}")
    logger.info(f"  Avg Inference Time: {metrics.get('avg_inference_time', 0.0):.4f} seconds")
    logger.info(f"  Total Evaluation Time: {total_time:.2f} seconds")
    
    return results


def generate_comparison_table(all_results: List[Dict[str, Any]], save_path: str):
    """
    Generate comparison table from all results.
    
    Args:
        all_results: List of results from all methods
        save_path: Path to save the table
    """
    rows = []
    
    for result in all_results:
        method_name = result['method_name']
        metrics = result['metrics']
        
        # Check if this is a skeleton-based method
        is_skeleton_based = any(x in method_name.upper() for x in ['2DCNN', '3DCNN', 'VIT'])
        inference_note = ""
        if is_skeleton_based:
            inference_note = " (includes YOLO skeleton extraction)"
        
        row = {
            'Method': method_name,
            'Num Samples': result.get('num_videos', 0),
            'Accuracy': f"{metrics.get('accuracy', 0.0):.4f}",
            'Precision': f"{metrics.get('precision', 0.0):.4f}",
            'Recall': f"{metrics.get('recall', 0.0):.4f}",
            'F1-Score': f"{metrics.get('f1_score', 0.0):.4f}",
            'Specificity': f"{metrics.get('specificity', 0.0):.4f}",
            'ROC AUC': f"{metrics.get('roc_auc', 0.0):.4f}" if 'roc_auc' in metrics and not np.isnan(metrics.get('roc_auc', 0.0)) else "N/A",
            'PR AUC': f"{metrics.get('pr_auc', 0.0):.4f}" if 'pr_auc' in metrics and not np.isnan(metrics.get('pr_auc', 0.0)) else "N/A",
            'Avg Inference Time (s)': f"{metrics.get('avg_inference_time', 0.0):.4f}{inference_note}",
            'Total Inference Time (s)': f"{metrics.get('total_inference_time', 0.0):.2f}",
            'TP': metrics.get('true_positives', 0),
            'TN': metrics.get('true_negatives', 0),
            'FP': metrics.get('false_positives', 0),
            'FN': metrics.get('false_negatives', 0),
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Save as CSV
    csv_path = save_path.replace('.txt', '.csv')
    df.to_csv(csv_path, index=False)
    logger.info(f"\nComparison table saved to {csv_path}")
    
    # Save as formatted text
    with open(save_path, 'w') as f:
        f.write("=" * 100 + "\n")
        f.write("FALL DETECTION METHODS COMPARISON\n")
        f.write("=" * 100 + "\n\n")
        
        # Add confusion matrix section
        f.write("\n" + "=" * 100 + "\n")
        f.write("CONFUSION MATRICES\n")
        f.write("=" * 100 + "\n\n")
        
        for result in all_results:
            method_name = result['method_name']
            metrics = result.get('metrics', {})
            tp = metrics.get('true_positives', 0)
            tn = metrics.get('true_negatives', 0)
            fp = metrics.get('false_positives', 0)
            fn = metrics.get('false_negatives', 0)
            
            f.write(f"\n{method_name}:\n")
            f.write(f"                    Predicted\n")
            f.write(f"                  Non-Fall  Fall\n")
            f.write(f"Actual Non-Fall     {tn:4d}   {fp:4d}\n")
            f.write(f"        Fall         {fn:4d}   {tp:4d}\n")
            f.write(f"\n  TN (True Negatives):  {tn:4d} - Correctly predicted non-fall\n")
            f.write(f"  FP (False Positives): {fp:4d} - Incorrectly predicted fall (non-fall → fall)\n")
            f.write(f"  FN (False Negatives): {fn:4d} - Incorrectly predicted non-fall (fall → non-fall)\n")
            f.write(f"  TP (True Positives):   {tp:4d} - Correctly predicted fall\n")
            f.write("-" * 100 + "\n")
        
        f.write("\n" + "=" * 100 + "\n")
        f.write("COMPARISON TABLE\n")
        f.write("=" * 100 + "\n\n")
        f.write(df.to_string(index=False))
        f.write("\n\n")
        f.write("=" * 100 + "\n")
        f.write("Legend:\n")
        f.write("  TP: True Positives, TN: True Negatives, FP: False Positives, FN: False Negatives\n")
        f.write("=" * 100 + "\n")
    
    logger.info(f"Comparison table saved to {save_path}")
    
    return df


def main():
    parser = argparse.ArgumentParser(description="Evaluate all fall detection methods")
    parser.add_argument("--video_list", type=str, default=None,
                       help="Path to file with list of video paths with labels (tab-separated: path\tlabel). If None, will sample or use shared test set")
    parser.add_argument("--shared_test_set", type=str, default="evaluation/data/shared_test_set.txt",
                       help="Path to shared test set file (used if --video_list not provided)")
    parser.add_argument("--fall_base_dir", type=str,
                       default="/home/reza/Documents/Datasets/Fall/Fall",
                       help="Base directory containing fall videos (may have Raw_Video and Raw_Video_part2)")
    parser.add_argument("--non_fall_dir", type=str, default=None,
                       help="Directory containing non-fall videos (optional if --kth_dir is provided)")
    parser.add_argument("--kth_dir", type=str, default=None,
                       help="Base directory of KTH dataset (will be used as non-fall samples if provided)")
    parser.add_argument("--video_dir", type=str, default=None,
                       help="[Deprecated] Use --fall_base_dir instead. Directory containing fall videos")
    parser.add_argument("--num_samples", type=int, default=100,
                       help="Number of videos to sample (if video_list not provided)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--window_length", type=int, default=32,
                       help="Window length in frames")
    parser.add_argument("--overlap", type=float, default=0.5,
                       help="Overlap ratio between windows")
    parser.add_argument("--vlm_num_frames", type=int, default=1,
                       help="Number of frames to sample per window for VLM (default: 1)")
    parser.add_argument("--few_shot_classifier", type=str,
                       default="evaluation/checkpoints/few_shot_vlm_classifier.pt",
                       help="Path to few-shot VLM classifier")
    parser.add_argument("--cnn_checkpoints", type=str, nargs='+',
                       default=[],
                       help="Paths to CNN model checkpoints (format: model_type:path)")
    parser.add_argument("--results_dir", type=str,
                       default="evaluation/results",
                       help="Directory to save results")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use for inference")
    
    args = parser.parse_args()
    
    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Load or sample videos with labels
    video_paths = []
    true_labels = []
    
    if args.video_list and os.path.exists(args.video_list):
        logger.info(f"Loading video list from {args.video_list}")
        # Check if it's a labeled file (tab-separated) or just video paths
        with open(args.video_list, 'r') as f:
            lines = f.readlines()
            if lines and '\t' in lines[0]:
                # Labeled file
                for line in lines:
                    line = line.strip()
                    if line:
                        parts = line.split('\t')
                        if len(parts) == 2:
                            video_paths.append(parts[0])
                            true_labels.append(bool(int(parts[1])))
            else:
                # Just video paths - try to load labels from separate file
                video_paths = load_sampled_videos(args.video_list)
                labels_path = args.video_list.replace('.txt', '_labels.txt')
                if os.path.exists(labels_path):
                    with open(labels_path, 'r') as f:
                        true_labels = [bool(int(line.strip())) for line in f if line.strip()]
                else:
                    # Assume all are falls if no labels
                    true_labels = [True] * len(video_paths)
                    logger.warning("No labels found, assuming all videos are falls")
    elif args.shared_test_set and os.path.exists(args.shared_test_set):
        # Use shared test set (ensures all models use same samples)
        logger.info(f"Loading shared test set from {args.shared_test_set}")
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
        # Sample balanced dataset
        from evaluation.dataset_sampler import sample_balanced_dataset
        logger.info(f"Sampling balanced dataset: {args.num_samples} fall + {args.num_samples} non-fall")
        _, test_pairs = sample_balanced_dataset(
            args.fall_base_dir,
            non_fall_dir=args.non_fall_dir,
            kth_dir=args.kth_dir,
            num_fall=args.num_samples,
            num_non_fall=args.num_samples,
            seed=args.seed,
            single_person_only=True  # Use single-person dataset only (best fitted process)
        )
        video_paths = [v for v, _ in test_pairs]
        true_labels = [bool(l) for _, l in test_pairs]
        # Save test set
        test_list_path = os.path.join(args.results_dir, "test_videos.txt")
        with open(test_list_path, 'w') as f:
            for video_path, label in test_pairs:
                f.write(f"{video_path}\t{label}\n")
        logger.info(f"Saved test set to {test_list_path}")
    
    logger.info(f"Evaluating {len(video_paths)} videos ({sum(true_labels)} fall, {len(true_labels) - sum(true_labels)} non-fall)")
    logger.info(f"\n{'='*60}")
    logger.info("DATASET BREAKDOWN")
    logger.info(f"{'='*60}")
    
    # Count videos by source
    part1_fall = sum(1 for v, l in zip(video_paths, true_labels) if "Raw_Video" in v and "Raw_Video_part2" not in v and l)
    part2_fall = sum(1 for v, l in zip(video_paths, true_labels) if "Raw_Video_part2" in v and l)
    non_fall = sum(1 for l in true_labels if not l)
    
    # Count KTH videos if used
    kth_videos = sum(1 for v, l in zip(video_paths, true_labels) if not l and "KTH" in v or any(act in v for act in ["boxing", "handclapping", "handwaving", "jogging", "running", "walking"]))
    
    logger.info(f"Fall videos (single-person only): {part1_fall}")
    logger.info(f"Non-fall videos: {non_fall}")
    if kth_videos > 0:
        logger.info(f"  - KTH dataset videos: {kth_videos}")
    logger.info(f"Total videos: {len(video_paths)}")
    logger.info(f"{'='*60}\n")
    
    all_results = []
    
    # 1. Simple Fall Detection
    try:
        logger.info("\n" + "="*60)
        logger.info("1. Simple Fall Detection")
        logger.info("="*60)
        logger.info(f"  Evaluating on {len(video_paths)} videos ({sum(true_labels)} fall, {len(true_labels) - sum(true_labels)} non-fall)")
        simple_evaluator = SimpleFallEvaluator(
            window_length=args.window_length,
            overlap=args.overlap
        )
        simple_results = evaluate_method(
            simple_evaluator,
            video_paths,
            "Simple Fall Detection",
            true_labels
        )
        all_results.append(simple_results)
    except Exception as e:
        logger.error(f"Error evaluating Simple Fall Detection: {e}", exc_info=True)
    
    # All videos are already single-person (from sample_balanced_dataset with single_person_only=True)
    vlm_video_paths = video_paths
    vlm_true_labels = true_labels
    logger.info(f"Using {len(vlm_video_paths)} videos for VLM evaluation (single-person dataset only)")
    
    # 2. Zero-Shot VLM
    try:
        logger.info("\n" + "="*60)
        logger.info("2. Zero-Shot VLM")
        logger.info("="*60)
        logger.info(f"  Evaluating on {len(vlm_video_paths)} videos ({sum(vlm_true_labels)} fall, {len(vlm_true_labels) - sum(vlm_true_labels)} non-fall)")
        logger.info(f"  Note: Only using single-person dataset (Part1) for fall videos")
        zero_shot_evaluator = ZeroShotVLMEvaluator(
            window_length=args.window_length,
            overlap=args.overlap,
            num_frames=args.vlm_num_frames,
            device=args.device
        )
        zero_shot_results = evaluate_method(
            zero_shot_evaluator,
            vlm_video_paths,
            "Zero-Shot VLM",
            vlm_true_labels
        )
        all_results.append(zero_shot_results)
    except Exception as e:
        logger.error(f"Error evaluating Zero-Shot VLM: {e}", exc_info=True)
    
    # 3. Few-Shot VLM
    if os.path.exists(args.few_shot_classifier):
        try:
            logger.info("\n" + "="*60)
            logger.info("3. Few-Shot VLM")
            logger.info("="*60)
            logger.info(f"  Evaluating on {len(vlm_video_paths)} videos ({sum(vlm_true_labels)} fall, {len(vlm_true_labels) - sum(vlm_true_labels)} non-fall)")
            logger.info(f"  Note: Only using single-person dataset (Part1) for fall videos")
            few_shot_evaluator = FewShotVLMEvaluator(
                classifier_path=args.few_shot_classifier,
                window_length=args.window_length,
                overlap=args.overlap,
                num_frames=args.vlm_num_frames,
                device=args.device
            )
            few_shot_results = evaluate_method(
                few_shot_evaluator,
                vlm_video_paths,
                "Few-Shot VLM",
                vlm_true_labels
            )
            all_results.append(few_shot_results)
        except Exception as e:
            logger.error(f"Error evaluating Few-Shot VLM: {e}", exc_info=True)
    else:
        logger.warning(f"Few-shot classifier not found at {args.few_shot_classifier}. Skipping.")
    
    # 4. CNN Models
    if args.cnn_checkpoints:
        for checkpoint_spec in args.cnn_checkpoints:
            try:
                # Parse checkpoint spec: "model_type:path"
                if ':' in checkpoint_spec:
                    model_type, checkpoint_path = checkpoint_spec.split(':', 1)
                else:
                    # Try to infer model type from path
                    checkpoint_path = checkpoint_spec
                    if '2dcnn' in checkpoint_path.lower():
                        model_type = '2dcnn_resnet'
                    elif '3dcnn' in checkpoint_path.lower():
                        model_type = '3dcnn_simple'
                    elif 'vit' in checkpoint_path.lower():
                        model_type = 'vit'
                    else:
                        logger.warning(f"Could not infer model type from {checkpoint_path}. Skipping.")
                        continue
                
                if not os.path.exists(checkpoint_path):
                    logger.warning(f"Checkpoint not found: {checkpoint_path}. Skipping.")
                    continue
                
                logger.info(f"\n{'='*60}")
                logger.info(f"4. {model_type.upper()}")
                logger.info(f"{'='*60}")
                logger.info(f"  Evaluating on {len(video_paths)} videos ({sum(true_labels)} fall, {len(true_labels) - sum(true_labels)} non-fall)")
                logger.info(f"  Note: Using all datasets (Part1 + Part2) for fall videos")
                logger.info(f"  Note: Inference time includes skeleton extraction from YOLO")
                
                cnn_evaluator = CNNEvaluator(
                    checkpoint_path=checkpoint_path,
                    model_type=model_type,
                    window_length=args.window_length,
                    overlap=args.overlap,
                    device=args.device
                )
                cnn_results = evaluate_method(
                    cnn_evaluator,
                    video_paths,
                    f"{model_type.upper()}",
                    true_labels
                )
                all_results.append(cnn_results)
            except Exception as e:
                logger.error(f"Error evaluating {checkpoint_spec}: {e}", exc_info=True)
    
    # Generate comparison table
    if len(all_results) > 0:
        logger.info("\n" + "="*60)
        logger.info("Generating Comparison Table")
        logger.info("="*60)
        
        table_path = os.path.join(args.results_dir, "comparison_table.txt")
        df = generate_comparison_table(all_results, table_path)
        
        # Save detailed results as JSON
        json_path = os.path.join(args.results_dir, "detailed_results.json")
        # Convert to JSON-serializable format
        json_results = []
        for result in all_results:
            json_result = {
                'method_name': result['method_name'],
                'num_videos': result['num_videos'],
                'metrics': result['metrics'],
                'total_evaluation_time': result['total_evaluation_time']
            }
            json_results.append(json_result)
        
        with open(json_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        logger.info(f"Detailed results saved to {json_path}")
        
        # Print summary
        print("\n" + "="*100)
        print("EVALUATION SUMMARY")
        print("="*100)
        print(df.to_string(index=False))
        print("="*100)
    else:
        logger.error("No results to compare!")


if __name__ == "__main__":
    main()

