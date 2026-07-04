"""
Anomaly detection module for fall detection using out-of-distribution detection.

This module implements anomaly detection by treating falls as out-of-distribution
events when models are trained only on normal activities (KTH dataset).
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple, List
from sklearn.metrics import roc_curve, auc
import math


def compute_confidence_score(softmax_probs: np.ndarray) -> np.ndarray:
    """
    Compute confidence score from softmax probabilities.
    
    Confidence = max(softmax_probs)
    Low confidence indicates anomaly (out-of-distribution).
    
    Args:
        softmax_probs: Softmax probabilities of shape [B, num_classes] or [num_classes]
        
    Returns:
        Confidence scores of shape [B] or scalar
    """
    if len(softmax_probs.shape) == 1:
        return np.max(softmax_probs)
    return np.max(softmax_probs, axis=1)


def compute_entropy(softmax_probs: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """
    Compute prediction entropy from softmax probabilities.
    
    Entropy = -sum(p * log(p))
    High entropy indicates uncertainty/anomaly.
    
    Args:
        softmax_probs: Softmax probabilities of shape [B, num_classes] or [num_classes]
        eps: Small value to avoid log(0)
        
    Returns:
        Entropy scores of shape [B] or scalar
    """
    # Clip probabilities to avoid log(0)
    probs_clipped = np.clip(softmax_probs, eps, 1.0 - eps)
    
    if len(softmax_probs.shape) == 1:
        entropy = -np.sum(probs_clipped * np.log(probs_clipped))
    else:
        entropy = -np.sum(probs_clipped * np.log(probs_clipped), axis=1)
    
    return entropy


def compute_anomaly_score(
    softmax_probs: np.ndarray,
    method: str = "hybrid",
    confidence_weight: float = 0.6,
    entropy_weight: float = 0.4,
    num_classes: int = 6
) -> np.ndarray:
    """
    Compute anomaly score from softmax probabilities.
    
    Args:
        softmax_probs: Softmax probabilities of shape [B, num_classes] or [num_classes]
        method: Method to use - "confidence", "entropy", or "hybrid"
        confidence_weight: Weight for confidence component (hybrid method)
        entropy_weight: Weight for entropy component (hybrid method)
        num_classes: Number of classes (for entropy normalization)
        
    Returns:
        Anomaly scores of shape [B] or scalar (higher = more anomalous)
    """
    if method == "confidence":
        # Low confidence = high anomaly
        confidence = compute_confidence_score(softmax_probs)
        anomaly_score = 1.0 - confidence  # Invert: high = anomaly
        return anomaly_score
    
    elif method == "entropy":
        # High entropy = high anomaly
        entropy = compute_entropy(softmax_probs)
        max_entropy = math.log(num_classes)  # Maximum possible entropy
        anomaly_score = entropy / max_entropy  # Normalize to [0, 1]
        return anomaly_score
    
    elif method == "hybrid":
        # Combine confidence and entropy
        confidence = compute_confidence_score(softmax_probs)
        entropy = compute_entropy(softmax_probs)
        
        # Normalize both to [0, 1]
        confidence_score = 1.0 - confidence  # Invert: high = anomaly
        max_entropy = math.log(num_classes)
        entropy_score = entropy / max_entropy  # Normalize to [0, 1]
        
        # Weighted combination
        anomaly_score = confidence_weight * confidence_score + entropy_weight * entropy_score
        return anomaly_score
    
    else:
        raise ValueError(f"Unknown anomaly detection method: {method}")


def find_optimal_threshold(
    anomaly_scores: np.ndarray,
    labels: Optional[np.ndarray] = None,
    method: str = "percentile",
    percentile: float = 95.0
) -> float:
    """
    Find optimal threshold for anomaly detection.
    
    Args:
        anomaly_scores: Anomaly scores from validation set (all normal activities)
        labels: Optional true labels (0=normal, 1=fall) for ROC-based threshold finding
        method: Method to use - "percentile" or "roc"
        percentile: Percentile to use for threshold (e.g., 95th percentile)
        
    Returns:
        Optimal threshold value
    """
    if method == "percentile":
        # Use percentile of validation scores (all normal activities)
        threshold = np.percentile(anomaly_scores, percentile)
        return float(threshold)
    
    elif method == "roc":
        if labels is None:
            raise ValueError("Labels required for ROC-based threshold finding")
        
        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(labels, anomaly_scores)
        
        # Find optimal threshold (Youden's J statistic: max(tpr - fpr))
        optimal_idx = np.argmax(tpr - fpr)
        threshold = thresholds[optimal_idx]
        
        return float(threshold)
    
    else:
        raise ValueError(f"Unknown threshold finding method: {method}")


def detect_anomaly(
    softmax_probs: np.ndarray,
    threshold: float,
    method: str = "hybrid",
    confidence_weight: float = 0.6,
    entropy_weight: float = 0.4,
    num_classes: int = 6
) -> Dict:
    """
    Detect anomaly (fall) from model softmax probabilities.
    
    Args:
        softmax_probs: Softmax probabilities of shape [B, num_classes] or [num_classes]
        threshold: Anomaly threshold (scores >= threshold are anomalies)
        method: Method to use - "confidence", "entropy", or "hybrid"
        confidence_weight: Weight for confidence component (hybrid method)
        entropy_weight: Weight for entropy component (hybrid method)
        num_classes: Number of classes
        
    Returns:
        Dictionary with:
        - is_anomaly: bool or array of bools
        - anomaly_score: float or array of floats
        - threshold: float
        - confidence: float or array of floats (max softmax prob)
        - entropy: float or array of floats
    """
    # Compute anomaly score
    anomaly_score = compute_anomaly_score(
        softmax_probs,
        method=method,
        confidence_weight=confidence_weight,
        entropy_weight=entropy_weight,
        num_classes=num_classes
    )
    
    # Detect anomaly
    is_anomaly = anomaly_score >= threshold
    
    # Compute additional metrics
    confidence = compute_confidence_score(softmax_probs)
    entropy = compute_entropy(softmax_probs)
    
    result = {
        'is_anomaly': is_anomaly,
        'anomaly_score': anomaly_score,
        'threshold': threshold,
        'confidence': confidence,
        'entropy': entropy
    }
    
    return result


def compute_anomaly_scores_batch(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    method: str = "hybrid",
    confidence_weight: float = 0.6,
    entropy_weight: float = 0.4,
    num_classes: int = 6
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute anomaly scores for all samples in a dataloader.
    
    Args:
        model: Trained model (in eval mode)
        dataloader: DataLoader with validation/test data
        device: Device to run inference on
        method: Anomaly detection method
        confidence_weight: Weight for confidence component
        entropy_weight: Weight for entropy component
        num_classes: Number of classes
        
    Returns:
        Tuple of (anomaly_scores, labels) as numpy arrays
    """
    model.eval()
    all_anomaly_scores = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            skeletons = batch['skeleton'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            outputs = model(skeletons)
            probs = F.softmax(outputs, dim=1)
            probs_np = probs.cpu().numpy()
            
            # Compute anomaly scores
            anomaly_scores = compute_anomaly_score(
                probs_np,
                method=method,
                confidence_weight=confidence_weight,
                entropy_weight=entropy_weight,
                num_classes=num_classes
            )
            
            all_anomaly_scores.extend(anomaly_scores)
            all_labels.extend(labels.cpu().numpy())
    
    return np.array(all_anomaly_scores), np.array(all_labels)









