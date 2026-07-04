#!/usr/bin/env python3
"""
Command-line demo script for VLM-based fall detection.

This script demonstrates both zero-shot and few-shot fall detection capabilities.
"""

import argparse
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from vlm.vlm_model import VisionLanguageModel
from vlm.zero_shot_inference import zero_shot_fall_detection, get_fall_detection_result
from vlm.few_shot_train import few_shot_predict, load_classifier, LinearClassifier
from vlm.config import (
    load_config,
    get_device,
    get_clip_model_name,
    get_num_frames,
    get_zero_shot_prompts,
    get_few_shot_config,
    ensure_directories,
)
import torch
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_zero_shot_results(probabilities: dict, threshold: float = 0.5):
    """
    Print zero-shot detection results in a formatted way.
    
    Args:
        probabilities: Dictionary from zero_shot_fall_detection.
        threshold: Threshold for binary classification.
    """
    print("\n" + "="*60)
    print("ZERO-SHOT FALL DETECTION RESULTS")
    print("="*60)
    
    # Print probabilities for each prompt
    print("\nPrompt Probabilities:")
    print("-" * 60)
    for prompt, prob in probabilities.items():
        if prompt != "fall_probability":
            # Truncate long prompts for display
            display_prompt = prompt[:50] + "..." if len(prompt) > 50 else prompt
            print(f"  {display_prompt:53s} {prob:.3f}")
    
    # Print fall probability and decision
    fall_prob = probabilities.get("fall_probability", 0.0)
    is_fall = fall_prob >= threshold
    
    print("\n" + "-" * 60)
    print(f"Fall Probability: {fall_prob:.3f}")
    print(f"Threshold: {threshold:.3f}")
    print(f"Decision: {'FALL DETECTED' if is_fall else 'NO FALL'}")
    print("="*60 + "\n")


def print_few_shot_results(result: dict, threshold: float = 0.5):
    """
    Print few-shot detection results in a formatted way.
    
    Args:
        result: Dictionary from few_shot_predict.
        threshold: Threshold for binary classification.
    """
    print("\n" + "="*60)
    print("FEW-SHOT FALL DETECTION RESULTS")
    print("="*60)
    
    fall_prob = result.get("fall_probability", 0.0)
    is_fall = result.get("is_fall", False)
    
    print(f"\nFall Probability: {fall_prob:.3f}")
    print(f"Threshold: {threshold:.3f}")
    print(f"Decision: {'FALL DETECTED' if is_fall else 'NO FALL'}")
    print("="*60 + "\n")


def main():
    """Main entry point for the demo script."""
    parser = argparse.ArgumentParser(
        description="VLM-based Fall Detection Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Zero-shot detection
  python demo_fall_detector.py --video path/to/video.mp4 --mode zero_shot
  
  # Few-shot detection (requires trained classifier)
  python demo_fall_detector.py --video path/to/video.mp4 --mode few_shot --classifier checkpoints/classifier.pt
  
  # Custom config file
  python demo_fall_detector.py --video path/to/video.mp4 --mode zero_shot --config configs/my_vlm_config.yaml
        """
    )
    
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to input video file"
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["zero_shot", "few_shot"],
        default="zero_shot",
        help="Detection mode: zero_shot or few_shot (default: zero_shot)"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (default: configs/vlm_config.yaml)"
    )
    
    parser.add_argument(
        "--classifier",
        type=str,
        default=None,
        help="Path to trained classifier (required for few_shot mode)"
    )
    
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Classification threshold (default: from config)"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.video):
        logger.error(f"Video file not found: {args.video}")
        sys.exit(1)
    
    if args.mode == "few_shot" and args.classifier is None:
        logger.error("Few-shot mode requires --classifier argument")
        sys.exit(1)
    
    if args.mode == "few_shot" and not os.path.exists(args.classifier):
        logger.error(f"Classifier file not found: {args.classifier}")
        sys.exit(1)
    
    # Load configuration
    try:
        config = load_config(args.config)
        ensure_directories(config)
        logger.info(f"Loaded configuration from: {args.config or 'default'}")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)
    
    # Get device and model name
    device = get_device(config)
    model_name = get_clip_model_name(config)
    
    # Initialize VLM
    logger.info(f"Initializing CLIP model: {model_name}")
    logger.info(f"Using device: {device}")
    
    try:
        vlm = VisionLanguageModel(model_name=model_name, device=device)
    except Exception as e:
        logger.error(f"Failed to initialize VLM: {e}")
        sys.exit(1)
    
    # Run detection based on mode
    if args.mode == "zero_shot":
        logger.info("Running zero-shot detection...")
        
        try:
            # Get threshold from config or args
            threshold = args.threshold
            if threshold is None:
                threshold = 0.5  # Default threshold
            
            # Run zero-shot detection
            probabilities = zero_shot_fall_detection(
                video_path=args.video,
                vlm=vlm,
                config=config
            )
            
            # Print results
            print_zero_shot_results(probabilities, threshold=threshold)
            
        except Exception as e:
            logger.error(f"Zero-shot detection failed: {e}", exc_info=True)
            sys.exit(1)
    
    elif args.mode == "few_shot":
        logger.info("Running few-shot detection...")
        
        try:
            # Get threshold from config or args
            threshold = args.threshold
            if threshold is None:
                few_shot_config = get_few_shot_config(config)
                threshold = few_shot_config.get("threshold", 0.5)
            
            # Load classifier
            # We need to know the embedding dimension and hidden_dim
            # For now, assume standard CLIP base model and no hidden layer
            # Note: If you used a hidden layer during training, set hidden_dim accordingly
            embedding_dim = vlm.image_embed_dim
            hidden_dim = None  # Set to the hidden_dim used during training if applicable
            logger.info(f"Loading classifier (embedding_dim={embedding_dim}, hidden_dim={hidden_dim})...")
            
            classifier = load_classifier(
                load_path=args.classifier,
                input_dim=embedding_dim,
                hidden_dim=hidden_dim
            )
            classifier = classifier.to(device)
            
            # Run few-shot prediction
            result = few_shot_predict(
                video_path=args.video,
                vlm=vlm,
                classifier=classifier,
                threshold=threshold,
                config=config
            )
            
            # Print results
            print_few_shot_results(result, threshold=threshold)
            
        except Exception as e:
            logger.error(f"Few-shot detection failed: {e}", exc_info=True)
            sys.exit(1)
    
    logger.info("Detection completed successfully!")


if __name__ == "__main__":
    main()

