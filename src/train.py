"""
Training script for skeleton-based action recognition
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau
from tqdm import tqdm
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import load_config, merge_configs, set_seed, ensure_dir, get_device, plot_training_curves, count_parameters, plot_confusion_matrix
from dataset import get_dataloaders
from models import create_model
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from anomaly_detection import compute_anomaly_scores_batch, find_optimal_threshold


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc="Training")
    for batch in pbar:
        skeletons = batch['skeleton'].to(device)
        labels = batch['label'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(skeletons)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100 * correct / total:.2f}%'
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device, return_predictions=False):
    """Validate model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validating", leave=False)
        for batch in pbar:
            skeletons = batch['skeleton'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(skeletons)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if return_predictions:
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * correct / total:.2f}%'
            })
    
    epoch_loss = running_loss / len(val_loader) if len(val_loader) > 0 else 0.0
    epoch_acc = 100 * correct / total if total > 0 else 0.0
    
    if return_predictions:
        return epoch_loss, epoch_acc, np.array(all_predictions), np.array(all_labels)
    return epoch_loss, epoch_acc


def train(config_path: str, model_type: str = None, skeleton_source: str = None,
         epochs: int = None, batch_size: int = None):
    """
    Main training function.
    
    Args:
        config_path: Path to base config file
        model_type: Override model type from config
        skeleton_source: Override skeleton source from config
        epochs: Override number of epochs from config
        batch_size: Override batch size from config
    """
    # Load configurations
    base_config = load_config(config_path)
    
    # If the provided config doesn't have 'dataset' section, it's likely a training-only config
    # In this case, load base_config.yaml first, then merge the provided config
    if 'dataset' not in base_config:
        base_config_path = "configs/base_config.yaml"
        if os.path.exists(base_config_path):
            print(f"Note: Provided config lacks 'dataset' section. Loading base config from {base_config_path}")
            base_config_full = load_config(base_config_path)
            # Merge: base_config_full (has dataset/model) + provided config (has training)
            base_config = merge_configs(base_config_full, base_config)
        else:
            raise ValueError(f"Config file {config_path} doesn't have 'dataset' section and base_config.yaml not found!")
    
    # Load training config (if not already merged from the provided config)
    # Only load default training_config.yaml if the provided config doesn't have training settings
    if 'training' not in base_config or not base_config.get('training'):
        training_config_path = "configs/training_config.yaml"
        if os.path.exists(training_config_path):
            training_config = load_config(training_config_path)
            base_config = merge_configs(base_config, training_config)
    
    # Load model-specific config
    model_type_config = model_type or base_config.get('model', {}).get('type', '3dcnn_simple')
    model_config_path = f"configs/model_configs/{model_type_config}.yaml"
    
    if os.path.exists(model_config_path):
        model_config = load_config(model_config_path)
        base_config = merge_configs(base_config, model_config)
    
    # Ensure 'training' key exists
    if 'training' not in base_config:
        base_config['training'] = {}
    
    # Override with command-line arguments
    if model_type:
        base_config['model']['type'] = model_type
    if skeleton_source:
        base_config['model']['skeleton_source'] = skeleton_source
    if epochs:
        base_config['training']['num_epochs'] = epochs
    if batch_size:
        base_config['training']['batch_size'] = batch_size
    
    # Set seed
    seed = base_config.get('seed', 42)
    set_seed(seed)
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Create directories
    checkpoints_dir = base_config.get('paths', {}).get('checkpoints_dir', './checkpoints')
    results_dir = base_config.get('paths', {}).get('results_dir', './results')
    ensure_dir(checkpoints_dir)
    ensure_dir(results_dir)
    
    # Get model configuration
    model_type = base_config['model']['type']
    skeleton_source = base_config['model']['skeleton_source']
    num_classes = base_config['dataset']['num_classes']
    
    print(f"\nTraining {model_type} model on {skeleton_source} skeletons")
    print("=" * 60)
    
    # Create data loaders
    print("Loading data...")
    try:
        train_loader, val_loader, test_loader = get_dataloaders(base_config)
    except ValueError as e:
        print(f"\n❌ Error loading data:")
        print(str(e))
        print("\n💡 Quick fix: Try using yolov11 skeleton source:")
        print(f"   python -m src.train --config configs/base_config.yaml --model_type {model_type} --skeleton_source yolov11")
        sys.exit(1)
    
    # Validate train loader exists
    if train_loader is None:
        raise ValueError("Training dataset is empty. Cannot proceed with training.")
    
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader) if val_loader else 0}, Test batches: {len(test_loader) if test_loader else 0}")
    
    # Check if validation set is empty (when combine_train_val=True)
    use_validation = val_loader is not None and len(val_loader) > 0
    if not use_validation:
        print("\nNote: Validation set is empty (train+val combined).")
        print("      Training will proceed without validation monitoring.")
        print("      Final evaluation will be performed on test set after training.")
    
    # Create model
    print(f"\nCreating {model_type} model...")
    # Extract model-specific parameters from config
    # Model config files contain documentation (layers, input_shape, etc.) that shouldn't be passed to model
    # Only pass actual model constructor parameters
    model_config = base_config.get('model', {})
    # Filter out non-model-constructor parameters (these are for documentation/reference only)
    excluded_keys = ['type', 'skeleton_source', 'name', 'input_shape', 'layers']
    model_kwargs = {k: v for k, v in model_config.items() 
                    if k not in excluded_keys and not isinstance(v, dict)}
    # Only pass simple parameters (not nested dicts like 'layers')
    model = create_model(model_type, num_classes=num_classes, **model_kwargs)
    model = model.to(device)
    
    # Print model info
    num_params = count_parameters(model)
    print(f"Model parameters: {num_params:,}")
    
    # Loss function
    training_config = base_config.get('training', {})
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    optimizer_type = training_config.get('optimizer', 'adam')
    learning_rate = training_config.get('learning_rate', 0.001)
    weight_decay = training_config.get('weight_decay', 0.0001)
    
    if optimizer_type.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_type.lower() == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_type.lower() == 'sgd':
        momentum = training_config.get('momentum', 0.9)
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")
    
    # Learning rate scheduler
    scheduler_config = training_config.get('scheduler', {})
    scheduler_type = scheduler_config.get('type', 'step')
    
    if scheduler_type == 'step':
        scheduler = StepLR(optimizer, step_size=scheduler_config.get('step_size', 10),
                          gamma=scheduler_config.get('gamma', 0.1))
    elif scheduler_type == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=training_config.get('num_epochs', 30))
    elif scheduler_type == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1,
                                     patience=scheduler_config.get('patience', 5))
    else:
        scheduler = None
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # Training loop
    num_epochs = training_config.get('num_epochs', 30)
    best_val_acc = 0.0
    best_epoch = 0
    
    print(f"\nStarting training for {num_epochs} epochs...")
    print("=" * 60)
    
    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        print("-" * 60)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate (if validation set exists)
        if use_validation:
            print("\nValidating...")
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            
            # Update learning rate
            if scheduler:
                if scheduler_type == 'plateau':
                    scheduler.step(val_loss)
                else:
                    scheduler.step()
            
            # Save history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            # Print epoch results
            print(f"\nTrain Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc:.2f}%")
            if scheduler:
                print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Save best model based on validation
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                
                # Add suffix for anomaly detection models to avoid confusion
                anomaly_config = base_config.get('anomaly_detection', {})
                suffix = "_anomaly" if anomaly_config.get('enabled', False) else ""
                checkpoint_name = f"{model_type}_{skeleton_source}{suffix}_best.pth"
                checkpoint_path = os.path.join(checkpoints_dir, checkpoint_name)
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss,
                    'config': base_config
                }, checkpoint_path)
                
                print(f"✓ Saved best model (val_acc: {val_acc:.2f}%) to {checkpoint_path}")
        else:
            # No validation set - update learning rate based on train loss
            if scheduler:
                if scheduler_type == 'plateau':
                    scheduler.step(train_loss)
                else:
                    scheduler.step()
            
            # Save history (no validation metrics)
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(0.0)
            history['val_acc'].append(0.0)
            
            # Print epoch results
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            if scheduler:
                print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Save model every epoch (or use train_acc for best model selection)
            if train_acc > best_val_acc:  # Reuse best_val_acc for train_acc tracking
                best_val_acc = train_acc
                best_epoch = epoch
                
                # Add suffix for anomaly detection models to avoid confusion
                anomaly_config = base_config.get('anomaly_detection', {})
                suffix = "_anomaly" if anomaly_config.get('enabled', False) else ""
                checkpoint_name = f"{model_type}_{skeleton_source}{suffix}_best.pth"
                checkpoint_path = os.path.join(checkpoints_dir, checkpoint_name)
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_acc': train_acc,
                    'train_loss': train_loss,
                    'config': base_config
                }, checkpoint_path)
                
                print(f"✓ Saved best model (train_acc: {train_acc:.2f}%) to {checkpoint_path}")
    
    print("\n" + "=" * 60)
    print(f"Training completed!")
    if use_validation:
        print(f"Best validation accuracy: {best_val_acc:.2f}% at epoch {best_epoch}")
    else:
        print(f"Best training accuracy: {best_val_acc:.2f}% at epoch {best_epoch}")
    
    # Evaluate on test set (final evaluation with comprehensive metrics)
    if test_loader is None or len(test_loader) == 0:
        print("\n" + "=" * 60)
        print("⚠ WARNING: Test set is empty. Skipping final evaluation.")
        print("=" * 60)
        print("Training completed successfully, but test evaluation was not possible.")
        return
    
    print("\n" + "=" * 60)
    print("FINAL EVALUATION ON TEST SET")
    print("=" * 60)
    
    # Get comprehensive evaluation results
    class_names = base_config['dataset']['actions']
    
    test_loss, test_acc, test_predictions, test_labels = validate(
        model, test_loader, criterion, device, return_predictions=True
    )
    
    # Calculate comprehensive metrics
    cm = confusion_matrix(test_labels, test_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_labels, test_predictions, average='weighted', zero_division=0
    )
    precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
        test_labels, test_predictions, average=None, zero_division=0
    )
    
    # Calculate per-class accuracy
    num_classes = len(class_names)
    per_class_acc = []
    for i in range(num_classes):
        mask = test_labels == i
        if mask.sum() > 0:
            per_class_acc.append((test_predictions[mask] == i).sum() / mask.sum())
        else:
            per_class_acc.append(0.0)
    per_class_acc = np.array(per_class_acc)
    
    # Ensure all arrays have the same length
    if len(precision_per_class) < num_classes:
        precision_per_class = np.pad(precision_per_class, (0, num_classes - len(precision_per_class)), 'constant')
    if len(recall_per_class) < num_classes:
        recall_per_class = np.pad(recall_per_class, (0, num_classes - len(recall_per_class)), 'constant')
    if len(f1_per_class) < num_classes:
        f1_per_class = np.pad(f1_per_class, (0, num_classes - len(f1_per_class)), 'constant')
    
    # Calculate top-k accuracy (if we have probabilities)
    # For now, we'll calculate it from the model outputs
    model.eval()
    top2_correct = 0
    top3_correct = 0
    total_topk = 0
    with torch.no_grad():
        for batch in test_loader:
            skeletons = batch['skeleton'].to(device)
            labels = batch['label'].to(device)
            outputs = model(skeletons)
            probs = torch.softmax(outputs, dim=1)
            
            # Top-2 accuracy
            _, top2_pred = torch.topk(probs, k=2, dim=1)
            top2_correct += (top2_pred == labels.unsqueeze(1)).any(dim=1).sum().item()
            
            # Top-3 accuracy
            _, top3_pred = torch.topk(probs, k=3, dim=1)
            top3_correct += (top3_pred == labels.unsqueeze(1)).any(dim=1).sum().item()
            
            total_topk += labels.size(0)
    
    top2_acc = 100 * top2_correct / total_topk if total_topk > 0 else 0.0
    top3_acc = 100 * top3_correct / total_topk if total_topk > 0 else 0.0
    
    # Print metrics
    print(f"\nTest Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
    print(f"Test Precision: {precision:.4f}, Test Recall: {recall:.4f}, Test F1: {f1:.4f}")
    print(f"Top-2 Accuracy: {top2_acc:.2f}%, Top-3 Accuracy: {top3_acc:.2f}%")
    print(f"\n✓ Final model performance on test set: {test_acc:.2f}%")
    
    # Print per-class metrics
    print(f"\nPer-Class Metrics:")
    print(f"{'Class':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 80)
    for i, class_name in enumerate(class_names):
        print(f"{class_name:<20} {per_class_acc[i]*100:<12.2f} {precision_per_class[i]:<12.4f} "
              f"{recall_per_class[i]:<12.4f} {f1_per_class[i]:<12.4f}")
    
    # Save confusion matrix
    cm_path = os.path.join(results_dir, f"{model_type}_{skeleton_source}_confusion_matrix.png")
    plot_confusion_matrix(cm, class_names, save_path=cm_path)
    print(f"\n✓ Confusion matrix saved to {cm_path}")
    
    # Save detailed metrics
    metrics_path = os.path.join(results_dir, f"{model_type}_{skeleton_source}_metrics.txt")
    with open(metrics_path, 'w') as f:
        f.write("FINAL EVALUATION METRICS\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Model: {model_type}\n")
        f.write(f"Skeleton Source: {skeleton_source}\n")
        f.write(f"Parameters: {num_params:,}\n\n")
        f.write(f"Overall Metrics:\n")
        f.write(f"  Accuracy:     {test_acc/100:.4f} ({test_acc:.2f}%)\n")
        f.write(f"  Top-2 Acc:    {top2_acc/100:.4f} ({top2_acc:.2f}%)\n")
        f.write(f"  Top-3 Acc:    {top3_acc/100:.4f} ({top3_acc:.2f}%)\n")
        f.write(f"  Loss:         {test_loss:.4f}\n")
        f.write(f"  Precision:    {precision:.4f}\n")
        f.write(f"  Recall:       {recall:.4f}\n")
        f.write(f"  F1-Score:     {f1:.4f}\n\n")
        f.write("Per-Class Metrics:\n")
        f.write("-" * 60 + "\n")
        for i, class_name in enumerate(class_names):
            f.write(f"{class_name}:\n")
            f.write(f"  Accuracy:  {per_class_acc[i]:.4f} ({per_class_acc[i]*100:.2f}%)\n")
            f.write(f"  Precision: {precision_per_class[i]:.4f}\n")
            f.write(f"  Recall:    {recall_per_class[i]:.4f}\n")
            f.write(f"  F1-Score:  {f1_per_class[i]:.4f}\n\n")
        f.write("\nConfusion Matrix:\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'':<15}")
        # Only print classes that appear in the confusion matrix
        num_classes_in_cm = cm.shape[0]
        for j in range(min(num_classes_in_cm, len(class_names))):
            f.write(f"{class_names[j][:10]:<12}")
        f.write("\n")
        for i in range(min(num_classes_in_cm, len(class_names))):
            f.write(f"{class_names[i][:14]:<15}")
            for j in range(min(num_classes_in_cm, len(class_names))):
                f.write(f"{cm[i, j]:<12}")
            f.write("\n")
    
    print(f"✓ Detailed metrics saved to {metrics_path}")
    
    # Plot training curves
    plot_path = os.path.join(results_dir, f"{model_type}_{skeleton_source}_training_curves.png")
    plot_training_curves(history, save_path=plot_path)
    
    # Save training history (including test metrics)
    history['test_loss'] = [test_loss]
    history['test_acc'] = [test_acc]
    history['test_top2_acc'] = [top2_acc]
    history['test_top3_acc'] = [top3_acc]
    history['test_precision'] = [precision]
    history['test_recall'] = [recall]
    history['test_f1'] = [f1]
    history['test_confusion_matrix'] = cm
    history['test_per_class_acc'] = per_class_acc
    history_path = os.path.join(results_dir, f"{model_type}_{skeleton_source}_history.npz")
    np.savez(history_path, **history)
    print(f"\nTraining history saved to {history_path}")
    print(f"  - Test accuracy: {test_acc:.2f}%")
    print(f"  - Test top-2 accuracy: {top2_acc:.2f}%")
    print(f"  - Test top-3 accuracy: {top3_acc:.2f}%")
    print(f"  - Test precision: {precision:.4f}")
    print(f"  - Test recall: {recall:.4f}")
    print(f"  - Test F1-score: {f1:.4f}")
    
    # Find anomaly detection threshold using validation set
    anomaly_config = base_config.get('anomaly_detection', {})
    if anomaly_config.get('enabled', False):
        print("\n" + "=" * 60)
        print("FINDING ANOMALY DETECTION THRESHOLD")
        print("=" * 60)
        
        # Get anomaly detection parameters
        method = anomaly_config.get('method', 'hybrid')
        confidence_weight = anomaly_config.get('confidence_weight', 0.6)
        entropy_weight = anomaly_config.get('entropy_weight', 0.4)
        threshold_method = anomaly_config.get('threshold_method', 'percentile')
        percentile = anomaly_config.get('percentile', 95.0)
        
        print(f"Anomaly detection method: {method}")
        print(f"Threshold finding method: {threshold_method}")
        
        # Use validation set to find threshold (all samples are normal activities)
        if use_validation and val_loader is not None and len(val_loader) > 0:
            print(f"\nComputing anomaly scores on validation set ({len(val_loader)} batches)...")
            val_anomaly_scores, val_labels = compute_anomaly_scores_batch(
                model, val_loader, device,
                method=method,
                confidence_weight=confidence_weight,
                entropy_weight=entropy_weight,
                num_classes=num_classes
            )
            
            # Find optimal threshold (validation set contains only normal activities)
            optimal_threshold = find_optimal_threshold(
                val_anomaly_scores,
                labels=None,  # All validation samples are normal
                method=threshold_method,
                percentile=percentile
            )
            
            print(f"\n✓ Optimal anomaly threshold: {optimal_threshold:.4f}")
            print(f"  Validation set statistics:")
            print(f"    Mean anomaly score: {np.mean(val_anomaly_scores):.4f}")
            print(f"    Std anomaly score: {np.std(val_anomaly_scores):.4f}")
            print(f"    Min anomaly score: {np.min(val_anomaly_scores):.4f}")
            print(f"    Max anomaly score: {np.max(val_anomaly_scores):.4f}")
            print(f"    Threshold ({percentile}th percentile): {optimal_threshold:.4f}")
            
            # Update config with threshold
            base_config.setdefault('anomaly_detection', {})['threshold'] = optimal_threshold
            
            # Reload best checkpoint and save with updated config
            suffix = "_anomaly" if anomaly_config.get('enabled', False) else ""
            checkpoint_name = f"{model_type}_{skeleton_source}{suffix}_best.pth"
            checkpoint_path = os.path.join(checkpoints_dir, checkpoint_name)
            
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                checkpoint['config'] = base_config
                checkpoint['anomaly_threshold'] = optimal_threshold
                torch.save(checkpoint, checkpoint_path)
                print(f"\n✓ Updated checkpoint with anomaly threshold: {checkpoint_path}")
            
            # Save threshold to results
            threshold_path = os.path.join(results_dir, f"{model_type}_{skeleton_source}_anomaly_threshold.txt")
            with open(threshold_path, 'w') as f:
                f.write("ANOMALY DETECTION THRESHOLD\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"Model: {model_type}\n")
                f.write(f"Skeleton Source: {skeleton_source}\n")
                f.write(f"Method: {method}\n")
                f.write(f"Threshold Finding Method: {threshold_method}\n")
                f.write(f"Percentile: {percentile}\n\n")
                f.write(f"Optimal Threshold: {optimal_threshold:.6f}\n\n")
                f.write("Validation Set Statistics:\n")
                f.write(f"  Mean: {np.mean(val_anomaly_scores):.6f}\n")
                f.write(f"  Std:  {np.std(val_anomaly_scores):.6f}\n")
                f.write(f"  Min:  {np.min(val_anomaly_scores):.6f}\n")
                f.write(f"  Max:  {np.max(val_anomaly_scores):.6f}\n")
            print(f"✓ Threshold details saved to {threshold_path}")
        else:
            print("\n⚠ Warning: No validation set available for threshold finding.")
            print("  Anomaly detection threshold will need to be set manually in config.")


def main():
    parser = argparse.ArgumentParser(description='Train skeleton action recognition model')
    parser.add_argument('--config', type=str, default='configs/base_config.yaml',
                       help='Path to base config file')
    parser.add_argument('--model_type', type=str, 
                       choices=['3dcnn_simple', '3dcnn_deep', '2dcnn_resnet', '2dcnn_lenet', '2dcnn', 'vit', 'stgcn', 'tcnt'],
                       help='Model type (overrides config)')
    parser.add_argument('--skeleton_source', type=str, choices=['openpose', 'yolov11'],
                       help='Skeleton source (overrides config)')
    parser.add_argument('--epochs', type=int, help='Number of epochs (overrides config)')
    parser.add_argument('--batch_size', type=int, help='Batch size (overrides config)')
    
    args = parser.parse_args()
    
    train(
        config_path=args.config,
        model_type=args.model_type,
        skeleton_source=args.skeleton_source,
        epochs=args.epochs,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()

