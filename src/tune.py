"""
Hyperparameter tuning script using Optuna
"""

import os
import sys
import argparse
import optuna
from optuna.trial import Trial
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
import yaml
import numpy as np
from tqdm import tqdm

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import load_config, merge_configs, set_seed, get_device
from dataset import get_dataloaders
from models import create_model


def objective(trial: Trial, config: dict, n_epochs: int = 20) -> float:
    """
    Optuna objective function for hyperparameter tuning.
    
    Args:
        trial: Optuna trial object
        config: Base configuration
        n_epochs: Number of epochs for each trial
        
    Returns:
        Best validation accuracy achieved
    """
    # Suggest hyperparameters
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32, 64])
    optimizer_name = trial.suggest_categorical('optimizer', ['adam', 'sgd', 'adamw'])
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    
    # Model-specific hyperparameters
    model_type = config['model']['type']
    model_kwargs = {}
    
    # Get model config defaults if available
    model_config = config.get('model', {}).get('architecture', {})
    
    if model_type == '3dcnn_simple':
        dropout = trial.suggest_float('dropout', 0.1, 0.7)
        model_kwargs = {'dropout': dropout}
        
    elif model_type == '3dcnn_deep':
        # Dense 3D CNN with 4 FC layers (memory optimized)
        dropout1 = trial.suggest_float('dropout1', 0.3, 0.7)
        dropout2 = trial.suggest_float('dropout2', 0.2, 0.6)
        dropout3 = trial.suggest_float('dropout3', 0.1, 0.5)
        dropout4 = trial.suggest_float('dropout4', 0.05, 0.3)
        model_kwargs = {
            'dropout1': dropout1,
            'dropout2': dropout2,
            'dropout3': dropout3,
            'dropout4': dropout4
        }
        
    elif model_type == '2dcnn_resnet':
        pretrained = trial.suggest_categorical('pretrained', [True, False])
        model_kwargs = {'pretrained': pretrained}
    elif model_type == '2dcnn_lenet':
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.7)
        weight_decay = trial.suggest_float('weight_decay', 1e-4, 1e-2, log=True)
        model_kwargs = {
            'dropout_rate': dropout_rate,
            'weight_decay': weight_decay
        }
    elif model_type == '2dcnn':
        # Backward compatibility: default to ResNet
        pretrained = trial.suggest_categorical('pretrained', [True, False])
        model_kwargs = {'pretrained': pretrained}
        
    elif model_type == 'vit':
        # ViT now uses 2D image input [B, 1, 32, 34]
        embed_dim = trial.suggest_categorical('embed_dim', [64, 128, 256])
        num_layers = trial.suggest_int('num_layers', 4, 8)  # Increased range for robustness
        num_heads = trial.suggest_categorical('num_heads', [4, 8, 16])  # More heads for robust attention
        dropout = trial.suggest_float('dropout', 0.1, 0.5)
        attention_dropout = trial.suggest_float('attention_dropout', 0.05, 0.2)
        mlp_ratio = trial.suggest_float('mlp_ratio', 2.0, 6.0)
        patch_size_h = trial.suggest_categorical('patch_size_h', [2, 4, 8])
        patch_size_w = trial.suggest_categorical('patch_size_w', [2, 4, 8])
        model_kwargs = {
            'img_size': (32, 34),  # 2D image size
            'patch_size': (patch_size_h, patch_size_w),
            'in_channels': 1,  # Single channel for 2D image
            'embed_dim': embed_dim,
            'num_layers': num_layers,
            'num_heads': num_heads,
            'dropout': dropout,
            'attention_dropout': attention_dropout,
            'mlp_ratio': mlp_ratio
        }
        
    elif model_type == 'stgcn':
        num_stages = trial.suggest_int('num_stages', 3, 6)
        base_channels = trial.suggest_categorical('base_channels', [32, 64, 128])
        dropout = trial.suggest_float('dropout', 0.1, 0.7)
        num_joints = model_config.get('num_joints', 17)
        in_channels = model_config.get('in_channels', 2)
        model_kwargs = {
            'num_joints': num_joints,
            'in_channels': in_channels,
            'num_stages': num_stages,
            'base_channels': base_channels,
            'dropout': dropout
        }
        
    elif model_type == 'tcnt':
        embed_dim = trial.suggest_categorical('embed_dim', [64, 128, 256])
        num_transformer_layers = trial.suggest_int('num_transformer_layers', 2, 6)
        num_heads = trial.suggest_categorical('num_heads', [2, 4, 8])
        mlp_ratio = trial.suggest_float('mlp_ratio', 2.0, 6.0)
        dropout = trial.suggest_float('dropout', 0.05, 0.3)
        # TCN channels - suggest first channel, others scale
        tcn_first = trial.suggest_categorical('tcn_first', [32, 64, 128])
        tcn_channels = [tcn_first, tcn_first * 2, tcn_first * 4]
        num_joints = model_config.get('num_joints', 17)
        in_channels = model_config.get('in_channels', 2)
        model_kwargs = {
            'num_joints': num_joints,
            'in_channels': in_channels,
            'embed_dim': embed_dim,
            'tcn_channels': tcn_channels,
            'num_transformer_layers': num_transformer_layers,
            'num_heads': num_heads,
            'mlp_ratio': mlp_ratio,
            'dropout': dropout
        }
    
    # Scheduler parameters
    scheduler_type = trial.suggest_categorical('scheduler', ['step', 'cosine', 'none'])
    if scheduler_type == 'step':
        step_size = trial.suggest_int('step_size', 5, 15)
        gamma = trial.suggest_float('gamma', 0.1, 0.5)
    
    # Ensure 'training' key exists
    if 'training' not in config:
        config['training'] = {}
    
    # Update config with suggested parameters
    config['training']['batch_size'] = batch_size
    config['training']['learning_rate'] = learning_rate
    config['training']['weight_decay'] = weight_decay
    config['training']['optimizer'] = optimizer_name
    
    # Set seed for reproducibility
    seed = config.get('seed', 42) + trial.number  # Different seed per trial
    set_seed(seed)
    
    # Get device
    device = get_device()
    
    # Create data loaders
    try:
        train_loader, val_loader, test_loader = get_dataloaders(config)
    except Exception as e:
        print(f"Error creating dataloaders: {e}")
        raise optuna.exceptions.TrialPruned()
    
    # Create model
    try:
        model = create_model(
            model_type,
            num_classes=config['dataset']['num_classes'],
            **model_kwargs
        )
        model = model.to(device)
    except Exception as e:
        print(f"Error creating model: {e}")
        import traceback
        traceback.print_exc()
        raise optuna.exceptions.TrialPruned()
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    if optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        momentum = 0.9
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, 
                            momentum=momentum, weight_decay=weight_decay)
    elif optimizer_name == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Scheduler
    if scheduler_type == 'step':
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_type == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)
    else:
        scheduler = None
    
    # Training loop
    best_val_acc = 0.0
    
    for epoch in range(n_epochs):
        # Train
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for batch in train_loader:
            skeletons = batch['skeleton'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(skeletons)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_acc = 100 * correct / total
        
        # Validate
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                skeletons = batch['skeleton'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(skeletons)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_acc = 100 * correct / total
        
        # Update best accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        
        # Update scheduler
        if scheduler:
            scheduler.step()
        
        # Report intermediate value
        trial.report(val_acc, epoch)
        
        # Handle pruning
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    
    return best_val_acc


def tune_hyperparameters(config_path: str, model_type: str = None, 
                        skeleton_source: str = None,
                        n_trials: int = 50, n_epochs: int = 20,
                        study_name: str = None):
    """
    Run hyperparameter tuning with Optuna.
    
    Args:
        config_path: Path to base config file
        model_type: Override model type from config
        skeleton_source: Override skeleton source from config
        n_trials: Number of trials to run
        n_epochs: Number of epochs per trial
        study_name: Name for the study (for resuming)
    """
    # Load configuration
    base_config = load_config(config_path)
    
    # Load training config
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
    
    model_type = base_config['model']['type']
    skeleton_source = base_config['model']['skeleton_source']
    
    # Create study name if not provided
    if study_name is None:
        study_name = f"{model_type}_{skeleton_source}_tuning"
    
    print(f"Starting hyperparameter tuning for {model_type} on {skeleton_source}")
    print(f"Study name: {study_name}")
    print(f"Number of trials: {n_trials}")
    print(f"Epochs per trial: {n_epochs}")
    print("=" * 60)
    
    # Create Optuna study
    study = optuna.create_study(
        study_name=study_name,
        direction='maximize',  # Maximize validation accuracy
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    )
    
    # Run optimization
    study.optimize(
        lambda trial: objective(trial, base_config.copy(), n_epochs),
        n_trials=n_trials,
        show_progress_bar=True
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("Optimization complete!")
    print("=" * 60)
    
    print(f"\nBest trial: {study.best_trial.number}")
    print(f"Best validation accuracy: {study.best_value:.2f}%")
    
    print("\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Save results
    results_dir = base_config.get('paths', {}).get('results_dir', './results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Save study to database
    study_db_path = os.path.join(results_dir, f"{study_name}.db")
    print(f"\nStudy saved to: sqlite:///{study_db_path}")
    
    # Save best params to YAML
    best_params_path = os.path.join(results_dir, f"{study_name}_best_params.yaml")
    with open(best_params_path, 'w') as f:
        yaml.dump(study.best_params, f)
    print(f"Best parameters saved to: {best_params_path}")
    
    # Generate optimization history plot
    try:
        import matplotlib.pyplot as plt
        
        fig = optuna.visualization.matplotlib.plot_optimization_history(study)
        plot_path = os.path.join(results_dir, f"{study_name}_optimization_history.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Optimization history plot saved to: {plot_path}")
        
        # Parameter importance plot
        fig = optuna.visualization.matplotlib.plot_param_importances(study)
        plot_path = os.path.join(results_dir, f"{study_name}_param_importances.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Parameter importance plot saved to: {plot_path}")
        
    except Exception as e:
        print(f"Could not generate plots: {e}")
    
    return study


def main():
    parser = argparse.ArgumentParser(description='Hyperparameter tuning with Optuna')
    parser.add_argument('--config', type=str, default='configs/base_config.yaml',
                       help='Path to base config file')
    parser.add_argument('--model_type', type=str, 
                       choices=['3dcnn_simple', '3dcnn_deep', '2dcnn_resnet', '2dcnn_lenet', '2dcnn', 'vit', 'stgcn', 'tcnt'],
                       help='Model type (overrides config)')
    parser.add_argument('--skeleton_source', type=str, choices=['openpose', 'yolov11'],
                       help='Skeleton source (overrides config)')
    parser.add_argument('--n_trials', type=int, default=50,
                       help='Number of trials')
    parser.add_argument('--n_epochs', type=int, default=20,
                       help='Epochs per trial')
    parser.add_argument('--study_name', type=str, default=None,
                       help='Study name (for resuming)')
    
    args = parser.parse_args()
    
    tune_hyperparameters(
        config_path=args.config,
        model_type=args.model_type,
        skeleton_source=args.skeleton_source,
        n_trials=args.n_trials,
        n_epochs=args.n_epochs,
        study_name=args.study_name
    )


if __name__ == "__main__":
    main()

