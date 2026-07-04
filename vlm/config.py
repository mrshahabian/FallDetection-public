"""
Configuration loader for VLM module.

Loads settings from YAML configuration files and provides easy access
to configuration parameters with sensible defaults.
"""

import yaml
import os
from typing import Dict, Any, List, Optional
import torch


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML config file. If None, uses default path.
        
    Returns:
        Dictionary containing configuration parameters.
    """
    if config_path is None:
        # Default to configs/vlm_config.yaml relative to project root
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(project_root, "configs", "vlm_config.yaml")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}\n"
            f"Please create the config file or specify a valid path."
        )
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Auto-detect device if set to "auto"
    if config.get("model", {}).get("device", "auto") == "auto":
        config["model"]["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    
    return config


def get_device(config: Dict[str, Any]) -> str:
    """
    Get device string from config.
    
    Args:
        config: Configuration dictionary.
        
    Returns:
        Device string ("cuda" or "cpu").
    """
    device = config.get("model", {}).get("device", "cpu")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return device


def get_clip_model_name(config: Dict[str, Any]) -> str:
    """
    Get CLIP model name from config.
    
    Args:
        config: Configuration dictionary.
        
    Returns:
        CLIP model identifier string.
    """
    return config.get("model", {}).get("clip_model_name", "openai/clip-vit-base-patch32")


def get_num_frames(config: Dict[str, Any]) -> int:
    """
    Get number of frames to sample per video.
    
    Args:
        config: Configuration dictionary.
        
    Returns:
        Number of frames.
    """
    return config.get("video", {}).get("num_frames", 8)


def get_zero_shot_prompts(config: Dict[str, Any]) -> List[str]:
    """
    Get zero-shot text prompts from config.
    
    Args:
        config: Configuration dictionary.
        
    Returns:
        List of prompt strings.
    """
    return config.get("zero_shot", {}).get("prompts", [
        "a person falling down",
        "a person standing normally",
    ])


def get_zero_shot_temperature(config: Dict[str, Any]) -> float:
    """
    Get temperature for zero-shot softmax.
    
    Args:
        config: Configuration dictionary.
        
    Returns:
        Temperature value.
    """
    return config.get("zero_shot", {}).get("temperature", 1.0)


def get_zero_shot_aggregation(config: Dict[str, Any]) -> str:
    """
    Get aggregation method for multi-frame videos.
    
    Args:
        config: Configuration dictionary.
        
    Returns:
        Aggregation method ("mean" or "max").
    """
    return config.get("zero_shot", {}).get("aggregation", "mean")


def get_few_shot_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get few-shot training configuration.
    
    Args:
        config: Configuration dictionary.
        
    Returns:
        Dictionary with few-shot training parameters.
    """
    few_shot_config = config.get("few_shot", {})
    return {
        "learning_rate": few_shot_config.get("learning_rate", 0.001),
        "epochs": few_shot_config.get("epochs", 50),
        "batch_size": few_shot_config.get("batch_size", 8),
        "optimizer": few_shot_config.get("optimizer", "adam"),
        "weight_decay": few_shot_config.get("weight_decay", 0.0001),
        "val_ratio": few_shot_config.get("val_ratio", 0.2),
        "early_stopping_patience": few_shot_config.get("early_stopping_patience", 10),
        "threshold": few_shot_config.get("threshold", 0.5),
    }


def get_paths(config: Dict[str, Any]) -> Dict[str, str]:
    """
    Get path configurations.
    
    Args:
        config: Configuration dictionary.
        
    Returns:
        Dictionary with path strings.
    """
    paths_config = config.get("paths", {})
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    return {
        "checkpoints_dir": os.path.join(project_root, paths_config.get("checkpoints_dir", "./vlm_checkpoints")),
        "results_dir": os.path.join(project_root, paths_config.get("results_dir", "./vlm_results")),
        "data_dir": os.path.join(project_root, paths_config.get("data_dir", "./vlm_data")),
        "fall_videos_dir": os.path.join(project_root, paths_config.get("fall_videos_dir", "./vlm_data/fall")),
        "non_fall_videos_dir": os.path.join(project_root, paths_config.get("non_fall_videos_dir", "./vlm_data/non_fall")),
    }


def ensure_directories(config: Dict[str, Any]) -> None:
    """
    Ensure all required directories exist.
    
    Args:
        config: Configuration dictionary.
    """
    paths = get_paths(config)
    for path in paths.values():
        os.makedirs(path, exist_ok=True)









