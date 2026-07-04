"""
Metrics calculation for fall detection evaluation.
"""

import numpy as np
from typing import Dict, List, Any
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, average_precision_score
)


def calculate_metrics(
    predictions: List[bool],
    labels: List[bool],
    probabilities: List[float] = None
) -> Dict[str, float]:
    """
    Calculate classification metrics.
    
    Args:
        predictions: List of predicted labels (True/False)
        labels: List of true labels (True/False)
        probabilities: List of prediction probabilities (optional)
    
    Returns:
        Dictionary with metrics
    """
    predictions = np.array(predictions, dtype=bool)
    labels = np.array(labels, dtype=bool)
    
    # Basic metrics
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, zero_division=0)
    recall = recall_score(labels, predictions, zero_division=0)
    f1 = f1_score(labels, predictions, zero_division=0)
    
    # Confusion matrix
    # Ensure we have both classes in labels for proper confusion matrix
    cm = confusion_matrix(labels, predictions, labels=[False, True])
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
    elif cm.size == 1:
        # Only one class present
        if labels[0] == False:
            tn, fp, fn, tp = (len(labels), 0, 0, 0) if predictions[0] == False else (0, len(labels), 0, 0)
        else:
            tn, fp, fn, tp = (0, 0, 0, len(labels)) if predictions[0] == True else (0, 0, len(labels), 0)
    else:
        tn, fp, fn, tp = (0, 0, 0, 0)
    
    # Additional metrics
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'specificity': float(specificity),
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn)
    }
    
    # ROC AUC and PR AUC if probabilities provided
    if probabilities is not None:
        try:
            # Check if we have both classes
            unique_labels = np.unique(labels)
            if len(unique_labels) > 1:
                roc_auc = roc_auc_score(labels, probabilities)
                pr_auc = average_precision_score(labels, probabilities)
                metrics['roc_auc'] = float(roc_auc)
                metrics['pr_auc'] = float(pr_auc)
            else:
                # Only one class present, AUC not defined
                metrics['roc_auc'] = float('nan')
                metrics['pr_auc'] = float('nan')
        except Exception as e:
            metrics['roc_auc'] = float('nan')
            metrics['pr_auc'] = float('nan')
    
    return metrics


def aggregate_window_results(
    per_window_results: List[Dict[str, Any]],
    true_label: bool = True
) -> Dict[str, Any]:
    """
    Aggregate results from multiple windows to video-level decision.
    
    Video-level decision: If majority of windows detect fall, video is considered fall.
    Example: 5 windows, 3 detect fall → video is fall
    
    Args:
        per_window_results: List of per-window results
        true_label: True label for the video (fall=True, non-fall=False)
    
    Returns:
        Aggregated results with video-level decision
    """
    if len(per_window_results) == 0:
        return {
            'fall_probability': 0.0,
            'is_fall': False,
            'num_windows': 0,
            'num_fall_windows': 0,
            'num_non_fall_windows': 0
        }
    
    # Get window-level predictions and probabilities
    fall_detections = [w.get('is_fall', False) for w in per_window_results]
    fall_probs = [w.get('fall_probability', 0.0) for w in per_window_results]
    
    # Count windows that detected fall
    num_fall_windows = sum(fall_detections)
    num_non_fall_windows = len(fall_detections) - num_fall_windows
    
    # Video-level decision: Majority vote
    # If >= 50% of windows detect fall, video is considered fall
    is_fall = num_fall_windows >= (len(fall_detections) / 2.0)
    
    # Average fall probability across all windows
    avg_fall_prob = np.mean(fall_probs)
    
    return {
        'fall_probability': float(avg_fall_prob),
        'is_fall': is_fall,
        'num_windows': len(per_window_results),
        'num_fall_windows': num_fall_windows,
        'num_non_fall_windows': num_non_fall_windows,
        'window_predictions': fall_detections,
        'window_probabilities': fall_probs
    }


def calculate_video_level_metrics(
    video_results: List[Dict[str, Any]],
    true_labels: List[bool] = None
) -> Dict[str, Any]:
    """
    Calculate metrics at video level.
    
    Args:
        video_results: List of video-level results from evaluators
        true_labels: Optional true labels (if None, assumes all are falls)
    
    Returns:
        Dictionary with aggregated metrics
    """
    if len(video_results) == 0:
        return {}
    
    # Extract predictions and probabilities
    predictions = [r.get('is_fall', False) for r in video_results]
    probabilities = [r.get('fall_probability', 0.0) for r in video_results]
    
    # Use true labels if provided, otherwise assume all are falls
    if true_labels is None:
        true_labels = [True] * len(video_results)
    
    # Calculate metrics
    metrics = calculate_metrics(predictions, true_labels, probabilities)
    
    # Add inference time statistics
    # Average inference time = average of ALL window inference times across ALL videos
    # This gives the true average inference time per window for the model (fair comparison)
    inference_times = []
    for r in video_results:
        times = r.get('inference_times', [])
        if times:
            inference_times.extend(times)  # Collect all window times from all videos
    
    if inference_times:
        # Average of all windows across all videos (fair comparison)
        metrics['avg_inference_time'] = float(np.mean(inference_times))
        metrics['std_inference_time'] = float(np.std(inference_times))
        metrics['total_inference_time'] = float(np.sum(inference_times))
        metrics['num_windows'] = len(inference_times)  # Total number of windows processed
    else:
        metrics['avg_inference_time'] = 0.0
        metrics['std_inference_time'] = 0.0
        metrics['total_inference_time'] = 0.0
        metrics['num_windows'] = 0
    
    return metrics

