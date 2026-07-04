"""
Utility functions for skeleton action recognition
"""

import os
import random
import yaml
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, Optional
import matplotlib.pyplot as plt
from pathlib import Path

try:
    from thop import profile, clever_format
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False
    print("Warning: thop not available. FLOPs calculation will be disabled.")


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load YAML configuration file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Dictionary containing configuration
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple configuration dictionaries.
    Later configs override earlier ones.
    
    Args:
        *configs: Variable number of config dictionaries
        
    Returns:
        Merged configuration dictionary
    """
    merged = {}
    for config in configs:
        merged = _deep_merge(merged, config)
    return merged


def _deep_merge(base: Dict, update: Dict) -> Dict:
    """Recursively merge two dictionaries."""
    for key, value in update.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            base[key] = _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def count_parameters(model: nn.Module) -> int:
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def calculate_flops(model: nn.Module, input_shape: Tuple[int, ...], 
                    device: str = "cuda") -> Tuple[float, str]:
    """
    Calculate FLOPs (Floating Point Operations) for a model.
    
    Args:
        model: PyTorch model
        input_shape: Input tensor shape (batch_size, ...)
        device: Device to run calculation on
        
    Returns:
        Tuple of (FLOPs as float, formatted string)
    """
    if not THOP_AVAILABLE:
        return 0.0, "N/A (thop not installed)"
    
    model.eval()
    dummy_input = torch.randn(input_shape).to(device)
    
    try:
        flops, params = profile(model, inputs=(dummy_input,), verbose=False)
        flops_formatted, _ = clever_format([flops, params], "%.3f")
        return flops, flops_formatted
    except Exception as e:
        print(f"Error calculating FLOPs: {e}")
        return 0.0, "Error"


def plot_training_curves(history: Dict[str, list], save_path: Optional[str] = None) -> None:
    """
    Plot training and validation loss/accuracy curves.
    
    Args:
        history: Dictionary with keys 'train_loss', 'val_loss', 'train_acc', 'val_acc'
        save_path: Optional path to save the plot
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Check if validation data exists (not all zeros)
    has_validation = any(v > 0 for v in history['val_loss']) or any(v > 0 for v in history['val_acc'])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    ax1.plot(epochs, history['train_loss'], 'b-', marker='o', label='Training Loss', linewidth=2, markersize=4)
    if has_validation:
        ax1.plot(epochs, history['val_loss'], 'r-', marker='s', label='Validation Loss', linewidth=2, markersize=4)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(epochs)
    
    # Accuracy plot
    ax2.plot(epochs, history['train_acc'], 'b-', marker='o', label='Training Accuracy', linewidth=2, markersize=4)
    if has_validation:
        ax2.plot(epochs, history['val_acc'], 'r-', marker='s', label='Validation Accuracy', linewidth=2, markersize=4)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(epochs)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_confusion_matrix(cm: np.ndarray, class_names: list, 
                         save_path: Optional[str] = None) -> None:
    """
    Plot confusion matrix with both normalized and unnormalized versions.
    
    Args:
        cm: Confusion matrix (numpy array)
        class_names: List of class names
        save_path: Optional path to save the plot
    """
    import seaborn as sns
    
    # Normalize confusion matrix (row-wise: percentage of actual class)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)  # Handle division by zero
    
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Left subplot: Unnormalized confusion matrix (counts)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                ax=ax1, cbar_kws={'label': 'Count'})
    ax1.set_xlabel('Predicted', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Actual', fontsize=12, fontweight='bold')
    ax1.set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
    
    # Right subplot: Normalized confusion matrix (percentages)
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=ax2, cbar_kws={'label': 'Percentage'})
    ax2.set_xlabel('Predicted', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Actual', fontsize=12, fontweight='bold')
    ax2.set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def get_device() -> torch.device:
    """Get available device (cuda or cpu)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

