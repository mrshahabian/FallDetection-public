#!/usr/bin/env python3
"""
Improve anomaly detection threshold using fall videos.

This script:
1. Processes fall videos and normal validation videos
2. Computes anomaly scores for both
3. Uses ROC curve to find optimal threshold
4. Updates checkpoint with new threshold
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'webapp'))

from models import create_model
from utils import load_config, get_device
from anomaly_detection import compute_anomaly_scores_batch
from dataset import get_dataloaders
from webapp.config import get_checkpoint_path
from webapp.video_processor import extract_with_yolov11
from preprocessing import generate_clips, create_2d_image_representation


def process_fall_videos(fall_videos_dir: str, model, device, model_type: str,
                       method: str = 'hybrid', confidence_weight: float = 0.6,
                       entropy_weight: float = 0.4, num_classes: int = 6):
    """
    Process fall videos and return anomaly scores.
    """
    from anomaly_detection import compute_anomaly_score
    from webapp.config import YOLOV11_MODEL_NAME, YOLOV11_CONFIDENCE
    
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    fall_videos = []
    for ext in video_extensions:
        fall_videos.extend(Path(fall_videos_dir).glob(f'*{ext}'))
        fall_videos.extend(Path(fall_videos_dir).glob(f'*{ext.upper()}'))
    
    if len(fall_videos) == 0:
        print(f"❌ No videos found in {fall_videos_dir}")
        return None
    
    print(f"Processing {len(fall_videos)} fall videos...")
    
    all_anomaly_scores = []
    model.eval()
    
    with torch.no_grad():
        for video_path in tqdm(fall_videos, desc="Processing fall videos"):
            try:
                # Extract skeleton
                extraction_result = extract_with_yolov11(
                    video_path=str(video_path),
                    model_name=YOLOV11_MODEL_NAME,
                    confidence=YOLOV11_CONFIDENCE
                )
                
                if isinstance(extraction_result, tuple):
                    keypoints, _ = extraction_result
                else:
                    keypoints = extraction_result
                
                if keypoints is None or (hasattr(keypoints, 'shape') and keypoints.shape[0] == 0):
                    continue
                
                # Generate clips
                clips = generate_clips(keypoints, clip_length=32, overlap=0.5)
                
                if len(clips) == 0:
                    continue
                
                # Process clips
                clip_scores = []
                for clip in clips:
                    # Preprocess based on model type
                    if model_type in ['3dcnn_simple', '3dcnn_deep']:
                        skeleton_tensor = torch.from_numpy(clip).float()
                        skeleton_tensor = skeleton_tensor.permute(2, 1, 0)  # (C, J, T)
                    elif model_type in ['2dcnn_resnet', '2dcnn_lenet', '2dcnn']:
                        image_2d = create_2d_image_representation(clip)
                        skeleton_tensor = torch.from_numpy(image_2d).float()
                        skeleton_tensor = skeleton_tensor.unsqueeze(0)
                    elif model_type == 'vit':
                        image_2d = create_2d_image_representation(clip)
                        skeleton_tensor = torch.from_numpy(image_2d).float()
                        skeleton_tensor = skeleton_tensor.unsqueeze(0)
                    elif model_type in ['stgcn', 'tcnt']:
                        skeleton_tensor = torch.from_numpy(clip).float()
                        skeleton_tensor = skeleton_tensor.permute(2, 1, 0)
                    
                    # Forward pass
                    clip_batch = skeleton_tensor.unsqueeze(0).to(device)
                    outputs = model(clip_batch)
                    probs = F.softmax(outputs, dim=1)
                    probs_np = probs.cpu().numpy()[0]
                    
                    # Compute anomaly score
                    anomaly_score = compute_anomaly_score(
                        probs_np,
                        method=method,
                        confidence_weight=confidence_weight,
                        entropy_weight=entropy_weight,
                        num_classes=num_classes
                    )
                    clip_scores.append(anomaly_score)
                
                # Average scores for this video
                if len(clip_scores) > 0:
                    all_anomaly_scores.append(np.mean(clip_scores))
            
            except Exception as e:
                print(f"\n⚠ Error processing {video_path}: {e}")
                continue
    
    return np.array(all_anomaly_scores) if len(all_anomaly_scores) > 0 else None


def find_optimal_threshold_roc(normal_scores: np.ndarray, fall_scores: np.ndarray):
    """
    Find optimal threshold using ROC curve.
    """
    # Create labels: 0 = normal, 1 = fall
    labels = np.concatenate([
        np.zeros(len(normal_scores)),  # Normal = 0
        np.ones(len(fall_scores))      # Fall = 1
    ])
    
    # Combine scores
    all_scores = np.concatenate([normal_scores, fall_scores])
    
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(labels, all_scores)
    roc_auc = auc(fpr, tpr)
    
    # Find optimal threshold (Youden's J statistic: max(tpr - fpr))
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    # Calculate metrics at optimal threshold
    tp = np.sum(fall_scores >= optimal_threshold)
    fp = np.sum(normal_scores >= optimal_threshold)
    tn = np.sum(normal_scores < optimal_threshold)
    fn = np.sum(fall_scores < optimal_threshold)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'threshold': optimal_threshold,
        'roc_auc': roc_auc,
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn
    }


def improve_threshold(checkpoint_path: str, fall_videos_dir: str, config_path: str = None,
                     model_type: str = None, output_dir: str = './results'):
    """
    Improve anomaly detection threshold using fall videos.
    """
    device = get_device()
    
    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get config
    if config_path:
        base_config = load_config(config_path)
        config = checkpoint.get('config', {})
        # Merge configs
        for key in base_config:
            if key not in config:
                config[key] = base_config[key]
    else:
        config = checkpoint.get('config', {})
    
    if not config:
        raise ValueError("No config found in checkpoint and no config_path provided")
    
    # Get model type
    if model_type is None:
        model_type = config.get('model', {}).get('type', '3dcnn_simple')
    
    num_classes = config.get('dataset', {}).get('num_classes', 6)
    
    # Get anomaly detection config
    anomaly_config = config.get('anomaly_detection', {})
    method = anomaly_config.get('method', 'hybrid')
    confidence_weight = anomaly_config.get('confidence_weight', 0.6)
    entropy_weight = anomaly_config.get('entropy_weight', 0.4)
    old_threshold = anomaly_config.get('threshold')
    
    print(f"\nCurrent anomaly detection config:")
    print(f"  Method: {method}")
    print(f"  Old threshold: {old_threshold:.4f}" if old_threshold else "  Old threshold: None")
    print(f"  Confidence weight: {confidence_weight}")
    print(f"  Entropy weight: {entropy_weight}")
    
    # Create model
    print(f"\nCreating {model_type} model...")
    model = create_model(model_type, num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Get normal activity scores from validation set
    print("\nComputing anomaly scores on validation set (normal activities)...")
    train_loader, val_loader, test_loader = get_dataloaders(config)
    
    if val_loader is None or len(val_loader) == 0:
        print("⚠ Warning: No validation set available. Using test set instead.")
        val_loader = test_loader
    
    normal_scores, _ = compute_anomaly_scores_batch(
        model, val_loader, device,
        method=method,
        confidence_weight=confidence_weight,
        entropy_weight=entropy_weight,
        num_classes=num_classes
    )
    
    print(f"  Normal activity scores: {len(normal_scores)} samples")
    print(f"  Mean: {np.mean(normal_scores):.4f}, Std: {np.std(normal_scores):.4f}")
    print(f"  Min: {np.min(normal_scores):.4f}, Max: {np.max(normal_scores):.4f}")
    
    # Get fall video scores
    print("\nComputing anomaly scores on fall videos...")
    fall_scores = process_fall_videos(
        fall_videos_dir, model, device, model_type,
        method, confidence_weight, entropy_weight, num_classes
    )
    
    if fall_scores is None or len(fall_scores) == 0:
        print("❌ No fall videos could be processed")
        return
    
    print(f"  Fall video scores: {len(fall_scores)} samples")
    print(f"  Mean: {np.mean(fall_scores):.4f}, Std: {np.std(fall_scores):.4f}")
    print(f"  Min: {np.min(fall_scores):.4f}, Max: {np.max(fall_scores):.4f}")
    
    # Find optimal threshold using ROC
    print("\nFinding optimal threshold using ROC curve...")
    roc_result = find_optimal_threshold_roc(normal_scores, fall_scores)
    
    optimal_threshold = roc_result['threshold']
    
    print(f"\n" + "=" * 60)
    print("OPTIMAL THRESHOLD RESULTS")
    print("=" * 60)
    
    # Handle case where threshold is inf or nan
    if not np.isfinite(optimal_threshold):
        print(f"⚠ WARNING: Optimal threshold is {optimal_threshold}")
        print(f"   This happens when fall scores are all 0.0 and normal scores are all > 0.0")
        print(f"   The model is too confident - threshold adjustment won't help.")
        print(f"   Recommendation: Use temperature scaling or fine-tune the model.")
        print(f"\n   Using fallback threshold: 0.0 (will detect all as anomalies)")
        optimal_threshold = 0.0
    else:
        print(f"Optimal threshold: {optimal_threshold:.6f}")
    print(f"ROC AUC: {roc_result['roc_auc']:.4f}")
    print(f"\nPerformance at optimal threshold:")
    print(f"  True Positives (Falls detected): {roc_result['tp']} / {len(fall_scores)}")
    print(f"  False Positives (Normal detected as fall): {roc_result['fp']} / {len(normal_scores)}")
    print(f"  True Negatives (Normal correctly identified): {roc_result['tn']} / {len(normal_scores)}")
    print(f"  False Negatives (Falls missed): {roc_result['fn']} / {len(fall_scores)}")
    print(f"\nMetrics:")
    print(f"  Precision: {roc_result['precision']:.4f}")
    print(f"  Recall: {roc_result['recall']:.4f}")
    print(f"  F1-Score: {roc_result['f1']:.4f}")
    
    if old_threshold:
        print(f"\nComparison with old threshold ({old_threshold:.6f}):")
        old_tp = np.sum(fall_scores >= old_threshold)
        old_fp = np.sum(normal_scores >= old_threshold)
        old_precision = old_tp / (old_tp + old_fp) if (old_tp + old_fp) > 0 else 0.0
        old_recall = old_tp / len(fall_scores) if len(fall_scores) > 0 else 0.0
        print(f"  Old - Precision: {old_precision:.4f}, Recall: {old_recall:.4f}")
        print(f"  New - Precision: {roc_result['precision']:.4f}, Recall: {roc_result['recall']:.4f}")
        print(f"  Improvement: Recall +{roc_result['recall'] - old_recall:.4f}")
    
    # Plot ROC curve (wrap in try-except to avoid crashing on plotting errors)
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        plt.figure(figsize=(10, 8))
        
        # ROC curve
        plt.subplot(2, 2, 1)
        plt.plot(roc_result['fpr'], roc_result['tpr'], 
                 label=f'ROC (AUC = {roc_result["roc_auc"]:.3f})', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Score distributions
        plt.subplot(2, 2, 2)
        # Handle case where all values are the same
        if len(np.unique(normal_scores)) > 1:
            plt.hist(normal_scores, bins=30, alpha=0.5, label='Normal', edgecolor='black')
        else:
            plt.bar([np.mean(normal_scores)], [len(normal_scores)], width=0.01, 
                    alpha=0.5, label=f'Normal (all={np.mean(normal_scores):.4f})', edgecolor='black')
        
        if len(np.unique(fall_scores)) > 1:
            plt.hist(fall_scores, bins=30, alpha=0.5, label='Fall', edgecolor='black')
        else:
            plt.bar([np.mean(fall_scores)], [len(fall_scores)], width=0.01, 
                    alpha=0.5, label=f'Fall (all={np.mean(fall_scores):.4f})', edgecolor='black')
        
        # Only plot threshold line if it's finite
        if np.isfinite(optimal_threshold):
            plt.axvline(optimal_threshold, color='r', linestyle='--', 
                        label=f'Optimal: {optimal_threshold:.4f}')
        if old_threshold and np.isfinite(old_threshold):
            plt.axvline(old_threshold, color='orange', linestyle='--', 
                        label=f'Old: {old_threshold:.4f}')
        plt.xlabel('Anomaly Score')
        plt.ylabel('Frequency')
        plt.title('Score Distributions')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Precision-Recall curve
        from sklearn.metrics import precision_recall_curve
        all_scores = np.concatenate([normal_scores, fall_scores])
        all_labels = np.concatenate([np.zeros(len(normal_scores)), np.ones(len(fall_scores))])
        pr_precision, pr_recall, pr_thresholds = precision_recall_curve(all_labels, all_scores)
        pr_auc = auc(pr_recall, pr_precision)
        
        plt.subplot(2, 2, 3)
        plt.plot(pr_recall, pr_precision, label=f'PR (AUC = {pr_auc:.3f})', linewidth=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Threshold vs Metrics
        plt.subplot(2, 2, 4)
        # Handle case where all values are the same or threshold is inf
        score_min = min(np.min(normal_scores), np.min(fall_scores))
        score_max = max(np.max(normal_scores), np.max(fall_scores))
        
        # If all scores are the same, create a small range around the value
        if score_min == score_max:
            thresholds_to_test = np.linspace(score_min - 0.1, score_max + 0.1, 100)
        else:
            thresholds_to_test = np.linspace(score_min, score_max, 100)
        precisions = []
        recalls = []
        f1s = []
        for t in thresholds_to_test:
            tp = np.sum(fall_scores >= t)
            fp = np.sum(normal_scores >= t)
            fn = np.sum(fall_scores < t)
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            precisions.append(p)
            recalls.append(r)
            f1s.append(f1)
        
        plt.plot(thresholds_to_test, precisions, label='Precision', linewidth=2)
        plt.plot(thresholds_to_test, recalls, label='Recall', linewidth=2)
        plt.plot(thresholds_to_test, f1s, label='F1', linewidth=2)
        # Only plot threshold line if it's finite and within range
        if np.isfinite(optimal_threshold) and score_min <= optimal_threshold <= score_max:
            plt.axvline(optimal_threshold, color='r', linestyle='--', 
                        label=f'Optimal: {optimal_threshold:.4f}')
        plt.xlabel('Threshold')
        plt.ylabel('Metric Value')
        plt.title('Metrics vs Threshold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f'{model_type}_threshold_improvement.png')
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"\n✓ ROC curve and analysis plots saved to {plot_path}")
    except Exception as e:
        print(f"\n⚠ Warning: Could not generate plots ({type(e).__name__}: {e})")
        print(f"   This is not critical - the analysis results above are still valid.")
        print(f"   The important information (threshold, metrics, recommendations) has been displayed.")
    
    # Update checkpoint (only if threshold is valid)
    if np.isfinite(optimal_threshold) and optimal_threshold >= 0:
        print(f"\nUpdating checkpoint with new threshold...")
        config['anomaly_detection']['threshold'] = float(optimal_threshold)
        checkpoint['config'] = config
        checkpoint['anomaly_threshold'] = float(optimal_threshold)
        
        # Save updated checkpoint
        updated_checkpoint_path = checkpoint_path.replace('.pth', '_improved.pth')
        if updated_checkpoint_path == checkpoint_path:
            updated_checkpoint_path = checkpoint_path.replace('_best.pth', '_improved_best.pth')
        
        torch.save(checkpoint, updated_checkpoint_path)
        print(f"✓ Updated checkpoint saved to {updated_checkpoint_path}")
        
        # Also update original if user wants
        print(f"\n💡 To use the improved threshold, either:")
        print(f"   1. Use the new checkpoint: {updated_checkpoint_path}")
        print(f"   2. Or update the original: {checkpoint_path}")
        print(f"      (Backup recommended before updating original)")
    else:
        print(f"\n⚠ Skipping checkpoint update - threshold is invalid ({optimal_threshold})")
        print(f"   The model needs to be fixed (temperature scaling or fine-tuning) before threshold can be improved.")
    
    # Save threshold details
    threshold_path = os.path.join(output_dir, f'{model_type}_improved_threshold.txt')
    with open(threshold_path, 'w') as f:
        f.write("IMPROVED ANOMALY DETECTION THRESHOLD\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Model: {model_type}\n")
        f.write(f"Method: {method}\n\n")
        f.write(f"Old Threshold: {old_threshold:.6f}\n" if old_threshold else "Old Threshold: None\n")
        f.write(f"New Threshold: {optimal_threshold:.6f}\n\n")
        f.write(f"ROC AUC: {roc_result['roc_auc']:.4f}\n")
        f.write(f"Precision: {roc_result['precision']:.4f}\n")
        f.write(f"Recall: {roc_result['recall']:.4f}\n")
        f.write(f"F1-Score: {roc_result['f1']:.4f}\n\n")
        f.write(f"Confusion Matrix:\n")
        f.write(f"  TP: {roc_result['tp']}, FP: {roc_result['fp']}\n")
        f.write(f"  TN: {roc_result['tn']}, FN: {roc_result['fn']}\n")
    
    print(f"✓ Threshold details saved to {threshold_path}")
    
    return optimal_threshold


def main():
    parser = argparse.ArgumentParser(description='Improve anomaly detection threshold using fall videos')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--fall_videos', type=str, required=True,
                       help='Directory containing fall videos')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file (if not in checkpoint)')
    parser.add_argument('--model_type', type=str, default=None,
                       help='Model type (if not in checkpoint)')
    parser.add_argument('--output_dir', type=str, default='./results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    improve_threshold(
        args.checkpoint,
        args.fall_videos,
        args.config,
        args.model_type,
        args.output_dir
    )


if __name__ == "__main__":
    main()

