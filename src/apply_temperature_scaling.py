#!/usr/bin/env python3
"""
Apply temperature scaling to reduce model overconfidence.

This script modifies the checkpoint to include temperature scaling,
which can help with anomaly detection when models are too confident.
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'webapp'))

from models import create_model
from utils import get_device
from anomaly_detection import compute_anomaly_score, detect_anomaly
from preprocessing import generate_clips, create_2d_image_representation
from webapp.video_processor import extract_with_yolov11
from webapp.config import YOLOV11_MODEL_NAME, YOLOV11_CONFIDENCE


def test_temperature_scaling(checkpoint_path: str, test_video_path: str, 
                             model_type: str = '3dcnn_simple',
                             temperatures: list = [1.0, 2.0, 3.0, 4.0, 5.0]):
    """
    Test different temperature values on a test video.
    """
    device = get_device()
    
    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    config = checkpoint.get('config', {})
    num_classes = config.get('dataset', {}).get('num_classes', 6)
    class_names = config.get('dataset', {}).get('actions', 
        ['walking', 'jogging', 'running', 'boxing', 'handwaving', 'handclapping'])
    
    anomaly_config = config.get('anomaly_detection', {})
    threshold = anomaly_config.get('threshold', 0.5)
    method = anomaly_config.get('method', 'hybrid')
    confidence_weight = anomaly_config.get('confidence_weight', 0.6)
    entropy_weight = anomaly_config.get('entropy_weight', 0.4)
    
    # Create model
    print(f"Creating {model_type} model...")
    model = create_model(model_type, num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Extract skeleton from test video
    print(f"\nExtracting skeleton from: {test_video_path}")
    extraction_result = extract_with_yolov11(
        video_path=test_video_path,
        model_name=YOLOV11_MODEL_NAME,
        confidence=YOLOV11_CONFIDENCE
    )
    
    if isinstance(extraction_result, tuple):
        keypoints, _ = extraction_result
    else:
        keypoints = extraction_result
    
    if keypoints is None or (hasattr(keypoints, 'shape') and keypoints.shape[0] == 0):
        print("❌ Could not extract skeleton from video")
        return
    
    # Generate clips
    clips = generate_clips(keypoints, clip_length=32, overlap=0.5)
    if len(clips) == 0:
        print("❌ No clips generated")
        return
    
    print(f"Generated {len(clips)} clips")
    print("\n" + "=" * 80)
    print("TEMPERATURE SCALING TEST RESULTS")
    print("=" * 80)
    print(f"{'Temp':<8} {'Confidence':<12} {'Entropy':<12} {'Anomaly Score':<15} {'Detected':<10} {'Top Class':<15}")
    print("-" * 80)
    
    # Test each temperature
    results = []
    with torch.no_grad():
        for temp in temperatures:
            all_probs = []
            all_confidences = []
            all_entropies = []
            all_anomaly_scores = []
            
            for clip in clips:
                # Preprocess
                if model_type in ['3dcnn_simple', '3dcnn_deep']:
                    skeleton_tensor = torch.from_numpy(clip).float()
                    skeleton_tensor = skeleton_tensor.permute(2, 1, 0)
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
                
                # Forward pass with temperature scaling
                clip_batch = skeleton_tensor.unsqueeze(0).to(device)
                outputs = model(clip_batch)
                probs = F.softmax(outputs / temp, dim=1)  # Temperature scaling
                probs_np = probs.cpu().numpy()[0]
                
                # Compute metrics
                from anomaly_detection import compute_confidence_score, compute_entropy
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
                all_confidences.append(confidence)
                all_entropies.append(entropy)
                all_anomaly_scores.append(anomaly_score)
            
            # Average
            avg_probs = np.mean(all_probs, axis=0)
            avg_confidence = np.mean(all_confidences)
            avg_entropy = np.mean(all_entropies)
            avg_anomaly_score = np.mean(all_anomaly_scores)
            top_class = class_names[np.argmax(avg_probs)]
            
            # Detect anomaly
            anomaly_result = detect_anomaly(
                avg_probs,
                threshold=threshold,
                method=method,
                confidence_weight=confidence_weight,
                entropy_weight=entropy_weight,
                num_classes=num_classes
            )
            
            detected = "YES" if anomaly_result['is_anomaly'] else "NO"
            
            print(f"{temp:<8.1f} {avg_confidence:<12.4f} {avg_entropy:<12.4f} "
                  f"{avg_anomaly_score:<15.4f} {detected:<10} {top_class:<15}")
            
            results.append({
                'temperature': temp,
                'confidence': avg_confidence,
                'entropy': avg_entropy,
                'anomaly_score': avg_anomaly_score,
                'detected': anomaly_result['is_anomaly'],
                'top_class': top_class,
                'probs': avg_probs
            })
    
    print("=" * 80)
    
    # Recommendations
    print("\nRECOMMENDATIONS:")
    print("-" * 80)
    
    # Find temperature that gives reasonable anomaly score
    good_temps = [r for r in results if r['anomaly_score'] > 0.1 and r['entropy'] > 0.1]
    
    if good_temps:
        best = max(good_temps, key=lambda x: x['anomaly_score'])
        print(f"✓ Recommended temperature: {best['temperature']:.1f}")
        print(f"  - Anomaly score: {best['anomaly_score']:.4f}")
        print(f"  - Entropy: {best['entropy']:.4f}")
        print(f"  - Fall detected: {best['detected']}")
    else:
        print("⚠ No temperature gives good anomaly scores")
        print("  Consider fine-tuning the model with fall videos")
    
    # Show probability distributions
    print("\nProbability distributions at different temperatures:")
    for r in results:
        if r['temperature'] in [1.0, 2.0, 3.0]:
            probs_str = ", ".join([f"{p:.3f}" for p in r['probs']])
            print(f"  Temp {r['temperature']:.1f}: [{probs_str}]")
    
    return results


def update_checkpoint_with_temperature(checkpoint_path: str, temperature: float, 
                                       output_path: str = None):
    """
    Update checkpoint to include temperature scaling parameter.
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Add temperature to config
    if 'config' not in checkpoint:
        checkpoint['config'] = {}
    
    if 'anomaly_detection' not in checkpoint['config']:
        checkpoint['config']['anomaly_detection'] = {}
    
    checkpoint['config']['anomaly_detection']['temperature'] = float(temperature)
    checkpoint['temperature'] = float(temperature)
    
    if output_path is None:
        output_path = checkpoint_path.replace('.pth', f'_temp{temperature:.1f}.pth')
        if output_path == checkpoint_path:
            output_path = checkpoint_path.replace('_best.pth', f'_temp{temperature:.1f}_best.pth')
    
    torch.save(checkpoint, output_path)
    print(f"✓ Updated checkpoint saved to: {output_path}")
    print(f"  Temperature: {temperature:.1f}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Test and apply temperature scaling')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--test_video', type=str, required=True,
                       help='Path to test video (fall video)')
    parser.add_argument('--model_type', type=str, default='3dcnn_simple',
                       help='Model type')
    parser.add_argument('--temperatures', type=float, nargs='+',
                       default=[1.0, 2.0, 3.0, 4.0, 5.0],
                       help='Temperature values to test')
    parser.add_argument('--apply', type=float, default=None,
                       help='Apply this temperature to checkpoint (creates new checkpoint)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for updated checkpoint')
    
    args = parser.parse_args()
    
    # Test temperatures
    results = test_temperature_scaling(
        args.checkpoint,
        args.test_video,
        args.model_type,
        args.temperatures
    )
    
    # Apply if requested
    if args.apply is not None:
        update_checkpoint_with_temperature(
            args.checkpoint,
            args.apply,
            args.output
        )


if __name__ == "__main__":
    main()









