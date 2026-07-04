#!/usr/bin/env python3
"""
Analyze model behavior on fall videos to understand why anomaly detection fails.

This script:
1. Processes fall videos to extract skeleton keypoints
2. Runs inference to get model predictions
3. Computes anomaly scores
4. Analyzes why falls are not being detected
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'webapp'))

from models import create_model
from utils import load_config, get_device
from anomaly_detection import compute_anomaly_score, detect_anomaly, compute_confidence_score, compute_entropy
from webapp.config import get_checkpoint_path
from webapp.video_processor import extract_with_yolov11
from preprocessing import generate_clips, create_2d_image_representation


def analyze_fall_video(video_path: str, model, device, model_type: str, threshold: float, 
                       method: str = 'hybrid', confidence_weight: float = 0.6, 
                       entropy_weight: float = 0.4, num_classes: int = 6):
    """
    Analyze a single fall video.
    
    Returns:
        Dictionary with analysis results
    """
    # Extract skeleton
    from webapp.config import YOLOV11_MODEL_NAME, YOLOV11_CONFIDENCE
    extraction_result = extract_with_yolov11(
        video_path=video_path,
        model_name=YOLOV11_MODEL_NAME,
        confidence=YOLOV11_CONFIDENCE
    )
    
    if isinstance(extraction_result, tuple):
        keypoints, _ = extraction_result
    else:
        keypoints = extraction_result
    
    if keypoints is None or (hasattr(keypoints, 'shape') and keypoints.shape[0] == 0):
        return None
    
    # Generate clips
    clips = generate_clips(keypoints, clip_length=32, overlap=0.5)
    
    if len(clips) == 0:
        return None
    
    # Process each clip
    all_probs = []
    all_anomaly_scores = []
    all_confidences = []
    all_entropies = []
    all_predictions = []
    
    model.eval()
    with torch.no_grad():
        for clip in clips:
            # Preprocess based on model type
            if model_type in ['3dcnn_simple', '3dcnn_deep']:
                skeleton_tensor = torch.from_numpy(clip).float()
                skeleton_tensor = skeleton_tensor.permute(2, 1, 0)  # (C, J, T)
            elif model_type in ['2dcnn_resnet', '2dcnn_lenet', '2dcnn']:
                image_2d = create_2d_image_representation(clip)
                skeleton_tensor = torch.from_numpy(image_2d).float()
                skeleton_tensor = skeleton_tensor.unsqueeze(0)  # (1, T, 2J)
            elif model_type == 'vit':
                image_2d = create_2d_image_representation(clip)
                skeleton_tensor = torch.from_numpy(image_2d).float()
                skeleton_tensor = skeleton_tensor.unsqueeze(0)  # (1, T, 2J)
            elif model_type in ['stgcn', 'tcnt']:
                skeleton_tensor = torch.from_numpy(clip).float()
                skeleton_tensor = skeleton_tensor.permute(2, 1, 0)  # (C, J, T)
            
            # Forward pass
            clip_batch = skeleton_tensor.unsqueeze(0).to(device)
            outputs = model(clip_batch)
            probs = F.softmax(outputs, dim=1)
            probs_np = probs.cpu().numpy()[0]
            
            # Debug: Check for NaN or extreme values
            if np.any(np.isnan(probs_np)) or np.any(probs_np < 0) or np.any(probs_np > 1):
                print(f"⚠ Warning: Invalid probabilities detected: {probs_np}")
            
            # Get prediction
            predicted_class = np.argmax(probs_np)
            
            # Compute metrics
            confidence = compute_confidence_score(probs_np)
            entropy = compute_entropy(probs_np)
            anomaly_score = compute_anomaly_score(
                probs_np,
                method=method,
                confidence_weight=confidence_weight,
                entropy_weight=entropy_weight,
                num_classes=num_classes
            )
            
            all_probs.append(probs_np)
            all_anomaly_scores.append(anomaly_score)
            all_confidences.append(confidence)
            all_entropies.append(entropy)
            all_predictions.append(predicted_class)
    
    # Average across clips
    avg_probs = np.mean(all_probs, axis=0)
    avg_anomaly_score = np.mean(all_anomaly_scores)
    avg_confidence = np.mean(all_confidences)
    avg_entropy = np.mean(all_entropies)
    most_common_prediction = max(set(all_predictions), key=all_predictions.count)
    
    # Detect anomaly
    anomaly_result = detect_anomaly(
        avg_probs,
        threshold=threshold,
        method=method,
        confidence_weight=confidence_weight,
        entropy_weight=entropy_weight,
        num_classes=num_classes
    )
    
    return {
        'video_path': video_path,
        'avg_probs': avg_probs,
        'predicted_class': most_common_prediction,
        'anomaly_score': avg_anomaly_score,
        'confidence': avg_confidence,
        'entropy': avg_entropy,
        'is_fall_detected': anomaly_result['is_anomaly'],
        'threshold': threshold,
        'per_clip_anomaly_scores': all_anomaly_scores,
        'per_clip_confidences': all_confidences,
        'per_clip_entropies': all_entropies,
        'per_clip_predictions': all_predictions
    }


def analyze_fall_videos(fall_videos_dir: str, checkpoint_path: str, 
                       model_type: str = '3dcnn_simple', output_dir: str = './results'):
    """
    Analyze all fall videos in a directory.
    """
    device = get_device()
    
    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get config
    config = checkpoint.get('config', {})
    num_classes = config.get('dataset', {}).get('num_classes', 6)
    class_names = config.get('dataset', {}).get('actions', ['walking', 'jogging', 'running', 'boxing', 'handwaving', 'handclapping'])
    
    # Get anomaly detection config
    anomaly_config = config.get('anomaly_detection', {})
    threshold = anomaly_config.get('threshold')
    method = anomaly_config.get('method', 'hybrid')
    confidence_weight = anomaly_config.get('confidence_weight', 0.6)
    entropy_weight = anomaly_config.get('entropy_weight', 0.4)
    
    if threshold is None:
        print("⚠ Warning: No threshold found in checkpoint. Using default 0.5")
        threshold = 0.5
    
    print(f"Anomaly detection config:")
    print(f"  Method: {method}")
    print(f"  Threshold: {threshold:.4f}")
    print(f"  Confidence weight: {confidence_weight}")
    print(f"  Entropy weight: {entropy_weight}")
    
    # Create model
    print(f"\nCreating {model_type} model...")
    model = create_model(model_type, num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Find fall videos
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    fall_videos = []
    for ext in video_extensions:
        fall_videos.extend(Path(fall_videos_dir).glob(f'*{ext}'))
        fall_videos.extend(Path(fall_videos_dir).glob(f'*{ext.upper()}'))
    
    if len(fall_videos) == 0:
        print(f"❌ No videos found in {fall_videos_dir}")
        return
    
    print(f"\nFound {len(fall_videos)} fall videos")
    print("=" * 60)
    
    # Analyze each video
    results = []
    for video_path in tqdm(fall_videos, desc="Analyzing videos"):
        try:
            result = analyze_fall_video(
                str(video_path), model, device, model_type,
                threshold, method, confidence_weight, entropy_weight, num_classes
            )
            if result:
                results.append(result)
        except Exception as e:
            print(f"\n⚠ Error analyzing {video_path}: {e}")
            import traceback
            traceback.print_exc()
    
    if len(results) == 0:
        print("❌ No videos could be analyzed")
        return
    
    # Print summary
    print("\n" + "=" * 60)
    print("ANALYSIS SUMMARY")
    print("=" * 60)
    
    anomaly_scores = [r['anomaly_score'] for r in results]
    confidences = [r['confidence'] for r in results]
    entropies = [r['entropy'] for r in results]
    fall_detected = [r['is_fall_detected'] for r in results]
    
    print(f"\nTotal videos analyzed: {len(results)}")
    print(f"Falls detected: {sum(fall_detected)} / {len(fall_detected)} ({100*sum(fall_detected)/len(fall_detected):.1f}%)")
    print(f"\nAnomaly Score Statistics:")
    print(f"  Mean: {np.mean(anomaly_scores):.4f}")
    print(f"  Std:  {np.std(anomaly_scores):.4f}")
    print(f"  Min:  {np.min(anomaly_scores):.4f}")
    print(f"  Max:  {np.max(anomaly_scores):.4f}")
    print(f"  Threshold: {threshold:.4f}")
    print(f"\nConfidence Statistics:")
    print(f"  Mean: {np.mean(confidences):.4f}")
    print(f"  Std:  {np.std(confidences):.4f}")
    print(f"  Min:  {np.min(confidences):.4f}")
    print(f"  Max:  {np.max(confidences):.4f}")
    print(f"\nEntropy Statistics:")
    print(f"  Mean: {np.mean(entropies):.4f}")
    print(f"  Std:  {np.std(entropies):.4f}")
    print(f"  Min:  {np.min(entropies):.4f}")
    print(f"  Max:  {np.max(entropies):.4f}")
    
    # Per-video details
    print(f"\n" + "=" * 60)
    print("PER-VIDEO DETAILS")
    print("=" * 60)
    print(f"{'Video':<40} {'Predicted':<15} {'Anomaly':<10} {'Conf':<8} {'Entropy':<8} {'Detected':<10}")
    print("-" * 100)
    for r in results:
        video_name = os.path.basename(r['video_path'])[:38]
        pred_class = class_names[r['predicted_class']]
        anomaly = r['anomaly_score']
        conf = r['confidence']
        entropy = r['entropy']
        detected = "YES" if r['is_fall_detected'] else "NO"
        print(f"{video_name:<40} {pred_class:<15} {anomaly:<10.4f} {conf:<8.4f} {entropy:<8.4f} {detected:<10}")
    
    # Save detailed results
    os.makedirs(output_dir, exist_ok=True)
    
    # Save CSV
    df_data = []
    for r in results:
        df_data.append({
            'video': os.path.basename(r['video_path']),
            'predicted_class': class_names[r['predicted_class']],
            'anomaly_score': r['anomaly_score'],
            'confidence': r['confidence'],
            'entropy': r['entropy'],
            'is_fall_detected': r['is_fall_detected'],
            'threshold': r['threshold']
        })
        # Add per-class probabilities
        for i, class_name in enumerate(class_names):
            df_data[-1][f'prob_{class_name}'] = r['avg_probs'][i]
    
    df = pd.DataFrame(df_data)
    csv_path = os.path.join(output_dir, f'{model_type}_fall_videos_analysis.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Detailed results saved to {csv_path}")
    
    # Plot distributions
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Anomaly score distribution
    # Handle case where all values are the same
    if len(np.unique(anomaly_scores)) > 1:
        axes[0, 0].hist(anomaly_scores, bins=20, edgecolor='black')
    else:
        # All values are the same, just show a single bar
        axes[0, 0].bar([np.mean(anomaly_scores)], [len(anomaly_scores)], width=0.01, edgecolor='black')
        axes[0, 0].text(np.mean(anomaly_scores), len(anomaly_scores), 
                       f'All values: {np.mean(anomaly_scores):.4f}', 
                       ha='center', va='bottom')
    axes[0, 0].axvline(threshold, color='r', linestyle='--', label=f'Threshold: {threshold:.4f}')
    axes[0, 0].set_xlabel('Anomaly Score')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Anomaly Score Distribution (Fall Videos)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Confidence distribution
    if len(np.unique(confidences)) > 1:
        axes[0, 1].hist(confidences, bins=20, edgecolor='black', color='orange')
    else:
        axes[0, 1].bar([np.mean(confidences)], [len(confidences)], width=0.01, edgecolor='black', color='orange')
        axes[0, 1].text(np.mean(confidences), len(confidences), 
                       f'All values: {np.mean(confidences):.4f}', 
                       ha='center', va='bottom')
    axes[0, 1].set_xlabel('Confidence')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Confidence Distribution (Fall Videos)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Entropy distribution
    if len(np.unique(entropies)) > 1:
        axes[1, 0].hist(entropies, bins=20, edgecolor='black', color='green')
    else:
        axes[1, 0].bar([np.mean(entropies)], [len(entropies)], width=0.01, edgecolor='black', color='green')
        axes[1, 0].text(np.mean(entropies), len(entropies), 
                       f'All values: {np.mean(entropies):.4f}', 
                       ha='center', va='bottom')
    axes[1, 0].set_xlabel('Entropy')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Entropy Distribution (Fall Videos)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Anomaly score vs Confidence scatter
    if len(np.unique(confidences)) > 1 or len(np.unique(anomaly_scores)) > 1:
        axes[1, 1].scatter(confidences, anomaly_scores, alpha=0.6)
    else:
        # All points are the same, show as a single point
        axes[1, 1].scatter([np.mean(confidences)], [np.mean(anomaly_scores)], 
                          s=100, alpha=0.6, label=f'All points: ({np.mean(confidences):.4f}, {np.mean(anomaly_scores):.4f})')
    axes[1, 1].axhline(threshold, color='r', linestyle='--', label=f'Threshold: {threshold:.4f}')
    axes[1, 1].set_xlabel('Confidence')
    axes[1, 1].set_ylabel('Anomaly Score')
    axes[1, 1].set_title('Anomaly Score vs Confidence')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'{model_type}_fall_videos_analysis.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"✓ Analysis plots saved to {plot_path}")
    
    # Debug: Print sample probabilities to understand what's happening
    print("\n" + "=" * 60)
    print("DEBUG: Sample Probabilities")
    print("=" * 60)
    if len(results) > 0:
        sample_result = results[0]
        print(f"Sample video: {os.path.basename(sample_result['video_path'])}")
        print(f"Average probabilities: {sample_result['avg_probs']}")
        print(f"Predicted class: {class_names[sample_result['predicted_class']]}")
        print(f"Anomaly score: {sample_result['anomaly_score']:.6f}")
        print(f"Confidence: {sample_result['confidence']:.6f}")
        print(f"Entropy: {sample_result['entropy']:.6f}")
        if len(sample_result.get('per_clip_anomaly_scores', [])) > 0:
            print(f"Per-clip anomaly scores (first 5): {sample_result['per_clip_anomaly_scores'][:5]}")
            print(f"Per-clip confidences (first 5): {sample_result['per_clip_confidences'][:5]}")
    
    # Recommendations
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    
    if np.mean(anomaly_scores) < threshold:
        print(f"\n⚠ ISSUE: Mean anomaly score ({np.mean(anomaly_scores):.4f}) is below threshold ({threshold:.4f})")
        print(f"   This is why falls are not being detected!")
        
        if np.mean(confidences) > 0.9:
            print(f"\n🔍 ROOT CAUSE: Model is too confident (mean confidence: {np.mean(confidences):.4f})")
            print(f"   When model is very confident, anomaly score becomes very low.")
            print(f"   The model is predicting normal activities with 100% confidence for all fall videos.")
            print(f"   This suggests the model has not learned to be uncertain about unseen patterns.")
            print(f"\n   Solutions:")
            print(f"   1. Fine-tune model with fall videos (best solution)")
            print(f"   2. Use temperature scaling to reduce overconfidence")
            print(f"   3. Lower threshold significantly (but will increase false positives)")
            print(f"   4. Try entropy-only method instead of hybrid")
        
        if np.mean(confidences) == 1.0 and np.mean(entropies) == 0.0:
            print(f"\n⚠ CRITICAL ISSUE: Model is perfectly confident (100%) on all fall videos!")
            print(f"   This means the model outputs exactly [1.0, 0.0, 0.0, 0.0, 0.0, 0.0] for all falls.")
            print(f"   This is a sign of severe overfitting or model collapse.")
            print(f"   The model needs to be retrained or fine-tuned with fall examples.")
        
        if np.max(anomaly_scores) < threshold:
            print(f"\n💡 SUGGESTION 1: Lower threshold to {np.max(anomaly_scores):.4f} (max fall score)")
            print(f"   This will detect all falls but may increase false positives")
        
        if np.mean(anomaly_scores) > 0:
            suggested_threshold = np.mean(anomaly_scores) - np.std(anomaly_scores)
            print(f"\n💡 SUGGESTION 2: Use mean - 1 std threshold: {suggested_threshold:.4f}")
            print(f"   This balances detection rate and false positives")
        
        print(f"\n💡 SUGGESTION 3: Use ROC curve to find optimal threshold")
        print(f"   Run: python src/improve_anomaly_threshold.py --fall_videos {fall_videos_dir}")
    
    else:
        print(f"\n✓ Mean anomaly score ({np.mean(anomaly_scores):.4f}) is above threshold ({threshold:.4f})")
        print(f"   Falls should be detected. Check individual video results above.")


def main():
    parser = argparse.ArgumentParser(description='Analyze fall videos to understand anomaly detection issues')
    parser.add_argument('--fall_videos', type=str, required=True,
                       help='Directory containing fall videos')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--model_type', type=str, default='3dcnn_simple',
                       help='Model type')
    parser.add_argument('--output_dir', type=str, default='./results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    analyze_fall_videos(
        args.fall_videos,
        args.checkpoint,
        args.model_type,
        args.output_dir
    )


if __name__ == "__main__":
    main()

