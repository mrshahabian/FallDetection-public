"""
Evaluation script for skeleton-based action recognition
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support
from tqdm import tqdm

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import load_config, merge_configs, get_device, calculate_flops, plot_confusion_matrix, count_parameters
from dataset import get_dataloaders
from models import create_model
from anomaly_detection import compute_anomaly_scores_batch, detect_anomaly, find_optimal_threshold
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve


def evaluate(model, test_loader, device, class_names):
    """
    Evaluate model on test set.
    
    Returns:
        Dictionary with predictions, labels, and metrics
    """
    model.eval()
    all_predictions = []
    all_labels = []
    all_probs = []
    all_file_paths = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            skeletons = batch['skeleton'].to(device)
            labels = batch['label'].to(device)
            file_paths = batch['file_path']
            
            outputs = model(skeletons)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_file_paths.extend(file_paths)
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='weighted', zero_division=0
    )
    
    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average=None, zero_division=0
    )
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    
    results = {
        'predictions': all_predictions,
        'labels': all_labels,
        'probs': all_probs,
        'file_paths': all_file_paths,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'f1_per_class': f1_per_class,
        'confusion_matrix': cm,
        'class_names': class_names
    }
    
    return results


def print_metrics(results):
    """Print evaluation metrics."""
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    print(f"\nOverall Metrics:")
    print(f"  Accuracy:  {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall:    {results['recall']:.4f}")
    print(f"  F1-Score:  {results['f1']:.4f}")
    
    print(f"\nPer-Class Metrics:")
    print(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 60)
    for i, class_name in enumerate(results['class_names']):
        print(f"{class_name:<20} {results['precision_per_class'][i]:<12.4f} "
              f"{results['recall_per_class'][i]:<12.4f} {results['f1_per_class'][i]:<12.4f}")


def save_predictions(results, save_path):
    """Save per-sample predictions to CSV."""
    df = pd.DataFrame({
        'file_path': results['file_paths'],
        'true_label': results['labels'],
        'predicted_label': results['predictions'],
        'true_class': [results['class_names'][l] for l in results['labels']],
        'predicted_class': [results['class_names'][p] for p in results['predictions']],
        'confidence': results['probs'].max(axis=1)
    })
    
    # Add per-class probabilities
    for i, class_name in enumerate(results['class_names']):
        df[f'prob_{class_name}'] = results['probs'][:, i]
    
    df.to_csv(save_path, index=False)
    print(f"\nPer-sample predictions saved to {save_path}")


def evaluate_model(checkpoint_path: str, config_path: str = None):
    """
    Evaluate a trained model.
    
    Args:
        checkpoint_path: Path to model checkpoint
        config_path: Optional path to config file (if not in checkpoint)
    """
    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Load config
    if 'config' in checkpoint:
        config = checkpoint['config']
    elif config_path:
        config = load_config(config_path)
    else:
        raise ValueError("Config not found in checkpoint and no config_path provided")
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Get model configuration
    model_type = config['model']['type']
    skeleton_source = config['model']['skeleton_source']
    num_classes = config['dataset']['num_classes']
    class_names = config['dataset']['actions']
    
    print(f"\nEvaluating {model_type} model trained on {skeleton_source} skeletons")
    print("=" * 60)
    
    # Create model
    print(f"\nCreating {model_type} model...")
    # Extract model-specific parameters from config (exclude type and skeleton_source)
    model_config = config.get('model', {})
    model_kwargs = {k: v for k, v in model_config.items() 
                    if k not in ['type', 'skeleton_source', 'name']}
    model = create_model(model_type, num_classes=num_classes, **model_kwargs)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    # Print model info
    num_params = count_parameters(model)
    print(f"Model parameters: {num_params:,}")
    
    # Calculate FLOPs
    print("\nCalculating FLOPs...")
    if model_type in ['3dcnn_simple', '3dcnn_deep', 'stgcn', 'tcnt']:
        input_shape = (1, 2, 17, 32)  # [B, C, J, T]
    elif model_type in ['2dcnn_resnet', '2dcnn_lenet', '2dcnn', 'vit']:
        input_shape = (1, 1, 32, 34)  # [B, C, H, W] (2D image)
    else:
        input_shape = (1, 2, 17, 32)
    
    flops, flops_formatted = calculate_flops(model, input_shape, device)
    print(f"FLOPs: {flops_formatted}")
    
    # Create data loaders
    print("\nLoading test data...")
    train_loader, val_loader, test_loader = get_dataloaders(config)
    print(f"Test batches: {len(test_loader)}")
    
    # Evaluate
    print("\nEvaluating on test set...")
    results = evaluate(model, test_loader, device, class_names)
    
    # Print metrics
    print_metrics(results)
    
    # Save results
    results_dir = config.get('paths', {}).get('results_dir', './results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Save predictions
    predictions_path = os.path.join(results_dir, f"{model_type}_{skeleton_source}_predictions.csv")
    save_predictions(results, predictions_path)
    
    # Save confusion matrix
    cm_path = os.path.join(results_dir, f"{model_type}_{skeleton_source}_confusion_matrix.png")
    plot_confusion_matrix(results['confusion_matrix'], class_names, save_path=cm_path)
    
    # Save metrics summary
    metrics_path = os.path.join(results_dir, f"{model_type}_{skeleton_source}_metrics.txt")
    with open(metrics_path, 'w') as f:
        f.write("EVALUATION METRICS\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Model: {model_type}\n")
        f.write(f"Skeleton Source: {skeleton_source}\n")
        f.write(f"Parameters: {num_params:,}\n")
        f.write(f"FLOPs: {flops_formatted}\n\n")
        f.write(f"Overall Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)\n")
        f.write(f"Overall Precision: {results['precision']:.4f}\n")
        f.write(f"Overall Recall: {results['recall']:.4f}\n")
        f.write(f"Overall F1-Score: {results['f1']:.4f}\n\n")
        f.write("Per-Class Metrics:\n")
        f.write("-" * 60 + "\n")
        for i, class_name in enumerate(class_names):
            f.write(f"{class_name}:\n")
            f.write(f"  Precision: {results['precision_per_class'][i]:.4f}\n")
            f.write(f"  Recall: {results['recall_per_class'][i]:.4f}\n")
            f.write(f"  F1-Score: {results['f1_per_class'][i]:.4f}\n\n")
    
    print(f"\nResults saved to {results_dir}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate skeleton action recognition model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file (if not in checkpoint)')
    
    args = parser.parse_args()
    
    evaluate_model(args.checkpoint, args.config)


def evaluate_anomaly_detection(
    model, test_loader, device, threshold: float,
    method: str = "hybrid", confidence_weight: float = 0.6,
    entropy_weight: float = 0.4, num_classes: int = 6,
    true_labels: Optional[np.ndarray] = None
):
    """
    Evaluate anomaly detection performance.
    
    Args:
        model: Trained model
        test_loader: DataLoader with test data
        device: Device to run inference on
        threshold: Anomaly detection threshold
        method: Anomaly detection method
        confidence_weight: Weight for confidence component
        entropy_weight: Weight for entropy component
        num_classes: Number of classes
        true_labels: Optional true binary labels (0=normal, 1=fall)
        
    Returns:
        Dictionary with anomaly detection metrics
    """
    model.eval()
    all_anomaly_scores = []
    all_predictions = []
    all_probs = []
    all_file_paths = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating anomaly detection"):
            skeletons = batch['skeleton'].to(device)
            file_paths = batch['file_path']
            
            outputs = model(skeletons)
            probs = torch.softmax(outputs, dim=1)
            probs_np = probs.cpu().numpy()
            
            # Detect anomalies
            for i in range(len(probs_np)):
                result = detect_anomaly(
                    probs_np[i],
                    threshold=threshold,
                    method=method,
                    confidence_weight=confidence_weight,
                    entropy_weight=entropy_weight,
                    num_classes=num_classes
                )
                all_anomaly_scores.append(result['anomaly_score'])
                all_predictions.append(1 if result['is_anomaly'] else 0)
                all_probs.append(probs_np[i])
                all_file_paths.append(file_paths[i])
    
    all_anomaly_scores = np.array(all_anomaly_scores)
    all_predictions = np.array(all_predictions)
    all_probs = np.array(all_probs)
    
    results = {
        'anomaly_scores': all_anomaly_scores,
        'predictions': all_predictions,
        'file_paths': all_file_paths,
        'threshold': threshold
    }
    
    # If we have true labels, compute metrics
    if true_labels is not None:
        from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, all_predictions, average='binary', zero_division=0
        )
        
        cm = confusion_matrix(true_labels, all_predictions)
        
        # Compute ROC curve
        fpr, tpr, _ = roc_curve(true_labels, all_anomaly_scores)
        roc_auc = auc(fpr, tpr)
        
        # Compute PR curve
        pr_precision, pr_recall, _ = precision_recall_curve(true_labels, all_anomaly_scores)
        pr_auc = auc(pr_recall, pr_precision)
        
        results.update({
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'fpr': fpr,
            'tpr': tpr,
            'pr_precision': pr_precision,
            'pr_recall': pr_recall
        })
    
    return results


def plot_roc_curve(fpr, tpr, roc_auc, save_path: str):
    """Plot ROC curve."""
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Anomaly Detection')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


def evaluate_model_with_anomaly_detection(
    checkpoint_path: str,
    config_path: str = None,
    test_labels_path: str = None
):
    """
    Evaluate a trained model with anomaly detection.
    
    Args:
        checkpoint_path: Path to model checkpoint
        config_path: Optional path to config file
        test_labels_path: Optional path to file with true binary labels (0=normal, 1=fall)
    """
    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Load config
    if 'config' in checkpoint:
        config = checkpoint['config']
    elif config_path:
        config = load_config(config_path)
    else:
        raise ValueError("Config not found in checkpoint and no config_path provided")
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Get model configuration
    model_type = config['model']['type']
    skeleton_source = config['model']['skeleton_source']
    num_classes = config['dataset']['num_classes']
    
    print(f"\nEvaluating {model_type} model with anomaly detection")
    print("=" * 60)
    
    # Check if anomaly detection is enabled
    anomaly_config = config.get('anomaly_detection', {})
    if not anomaly_config.get('enabled', False):
        print("⚠ Warning: Anomaly detection not enabled in config.")
        print("  Set anomaly_detection.enabled=true in config to use anomaly detection.")
        return
    
    # Get anomaly detection parameters
    method = anomaly_config.get('method', 'hybrid')
    confidence_weight = anomaly_config.get('confidence_weight', 0.6)
    entropy_weight = anomaly_config.get('entropy_weight', 0.4)
    threshold = anomaly_config.get('threshold')
    
    if threshold is None:
        print("⚠ Warning: Anomaly detection threshold not found in config.")
        print("  Run training with anomaly detection enabled to find threshold.")
        return
    
    print(f"Anomaly detection method: {method}")
    print(f"Threshold: {threshold:.4f}")
    
    # Create model
    print(f"\nCreating {model_type} model...")
    # Extract model-specific parameters from config (exclude type and skeleton_source)
    model_config = config.get('model', {})
    model_kwargs = {k: v for k, v in model_config.items() 
                    if k not in ['type', 'skeleton_source', 'name']}
    model = create_model(model_type, num_classes=num_classes, **model_kwargs)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    # Create data loaders
    print("\nLoading test data...")
    train_loader, val_loader, test_loader = get_dataloaders(config)
    print(f"Test batches: {len(test_loader)}")
    
    # Load true labels if provided
    true_labels = None
    if test_labels_path and os.path.exists(test_labels_path):
        true_labels = np.loadtxt(test_labels_path, dtype=int)
        print(f"Loaded {len(true_labels)} true labels from {test_labels_path}")
    
    # Evaluate anomaly detection
    print("\nEvaluating anomaly detection on test set...")
    results = evaluate_anomaly_detection(
        model, test_loader, device,
        threshold=threshold,
        method=method,
        confidence_weight=confidence_weight,
        entropy_weight=entropy_weight,
        num_classes=num_classes,
        true_labels=true_labels
    )
    
    # Print metrics
    print("\n" + "=" * 60)
    print("ANOMALY DETECTION RESULTS")
    print("=" * 60)
    
    print(f"\nThreshold: {results['threshold']:.4f}")
    print(f"Anomaly predictions: {np.sum(results['predictions'])} / {len(results['predictions'])}")
    print(f"Anomaly rate: {100 * np.mean(results['predictions']):.2f}%")
    
    if true_labels is not None:
        print(f"\nMetrics:")
        print(f"  Precision: {results['precision']:.4f}")
        print(f"  Recall:    {results['recall']:.4f}")
        print(f"  F1-Score:  {results['f1']:.4f}")
        print(f"  ROC AUC:   {results['roc_auc']:.4f}")
        print(f"  PR AUC:    {results['pr_auc']:.4f}")
        
        print(f"\nConfusion Matrix:")
        print(f"  Normal predicted as Normal: {results['confusion_matrix'][0, 0]}")
        print(f"  Normal predicted as Fall:  {results['confusion_matrix'][0, 1]}")
        print(f"  Fall predicted as Normal:  {results['confusion_matrix'][1, 0]}")
        print(f"  Fall predicted as Fall:    {results['confusion_matrix'][1, 1]}")
    
    # Save results
    results_dir = config.get('paths', {}).get('results_dir', './results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Save predictions
    predictions_path = os.path.join(results_dir, f"{model_type}_{skeleton_source}_anomaly_predictions.csv")
    import pandas as pd
    df = pd.DataFrame({
        'file_path': results['file_paths'],
        'anomaly_score': results['anomaly_scores'],
        'is_anomaly': results['predictions']
    })
    if true_labels is not None:
        df['true_label'] = true_labels
    df.to_csv(predictions_path, index=False)
    print(f"\n✓ Predictions saved to {predictions_path}")
    
    # Plot ROC curve if we have true labels
    if true_labels is not None:
        roc_path = os.path.join(results_dir, f"{model_type}_{skeleton_source}_anomaly_roc.png")
        plot_roc_curve(results['fpr'], results['tpr'], results['roc_auc'], roc_path)
        print(f"✓ ROC curve saved to {roc_path}")
    
    # Save metrics
    metrics_path = os.path.join(results_dir, f"{model_type}_{skeleton_source}_anomaly_metrics.txt")
    with open(metrics_path, 'w') as f:
        f.write("ANOMALY DETECTION EVALUATION METRICS\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Model: {model_type}\n")
        f.write(f"Skeleton Source: {skeleton_source}\n")
        f.write(f"Method: {method}\n")
        f.write(f"Threshold: {threshold:.6f}\n\n")
        f.write(f"Anomaly Rate: {100 * np.mean(results['predictions']):.2f}%\n")
        f.write(f"Total Samples: {len(results['predictions'])}\n")
        f.write(f"Anomalies Detected: {np.sum(results['predictions'])}\n\n")
        
        if true_labels is not None:
            f.write("Metrics:\n")
            f.write(f"  Precision: {results['precision']:.4f}\n")
            f.write(f"  Recall:    {results['recall']:.4f}\n")
            f.write(f"  F1-Score:  {results['f1']:.4f}\n")
            f.write(f"  ROC AUC:   {results['roc_auc']:.4f}\n")
            f.write(f"  PR AUC:    {results['pr_auc']:.4f}\n\n")
            f.write("Confusion Matrix:\n")
            f.write(f"  Normal -> Normal: {results['confusion_matrix'][0, 0]}\n")
            f.write(f"  Normal -> Fall:   {results['confusion_matrix'][0, 1]}\n")
            f.write(f"  Fall -> Normal:   {results['confusion_matrix'][1, 0]}\n")
            f.write(f"  Fall -> Fall:     {results['confusion_matrix'][1, 1]}\n")
    
    print(f"✓ Metrics saved to {metrics_path}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate skeleton action recognition model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file (if not in checkpoint)')
    parser.add_argument('--anomaly', action='store_true',
                       help='Evaluate with anomaly detection')
    parser.add_argument('--test_labels', type=str, default=None,
                       help='Path to file with true binary labels (0=normal, 1=fall)')
    
    args = parser.parse_args()
    
    if args.anomaly:
        evaluate_model_with_anomaly_detection(
            args.checkpoint,
            args.config,
            args.test_labels
        )
    else:
        evaluate_model(args.checkpoint, args.config)

