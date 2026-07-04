"""
Model inference module for web application
Handles model loading, caching, and inference for all models
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'vlm'))

from models import create_model
from utils import load_config
from anomaly_detection import detect_anomaly, compute_anomaly_score

# Import webapp config (must be after src imports to avoid conflicts)
sys.path.insert(0, os.path.dirname(__file__))
from config import (
    MODEL_CHECKPOINTS, CLASS_NAMES, DEVICE, CHECKPOINTS_DIR,
    get_checkpoint_path, ANOMALY_MODEL_CHECKPOINTS,
    FALL_DETECTION_CNN_CHECKPOINTS, FALL_DETECTION_CLASS_NAMES
)
from video_processor import extract_and_preprocess_video

# Import VLM modules
try:
    # Add parent directory to path so we can import vlm as a package
    parent_dir = os.path.dirname(os.path.dirname(__file__))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    
    # Import VLM modules as a package (handles relative imports)
    import vlm.vlm_model as vlm_model_module
    import vlm.zero_shot_inference as zero_shot_module
    import vlm.config as vlm_config_module
    
    VisionLanguageModel = vlm_model_module.VisionLanguageModel
    zero_shot_activity_recognition = zero_shot_module.zero_shot_activity_recognition
    zero_shot_fall_detection = zero_shot_module.zero_shot_fall_detection
    load_vlm_config = vlm_config_module.load_config
    
    VLM_AVAILABLE = True
    print("VLM modules loaded successfully")
except ImportError as e:
    print(f"Warning: VLM modules not available (ImportError): {e}")
    VLM_AVAILABLE = False
    load_vlm_config = None
    VisionLanguageModel = None
    zero_shot_activity_recognition = None
    zero_shot_fall_detection = None
except Exception as e:
    print(f"Warning: VLM modules not available (Exception): {e}")
    import traceback
    traceback.print_exc()
    VLM_AVAILABLE = False
    load_vlm_config = None
    VisionLanguageModel = None
    zero_shot_activity_recognition = None
    zero_shot_fall_detection = None


# Model cache
_model_cache = {}
_vlm_model = None
# Cache for class names per model (loaded from checkpoint)
_model_class_names = {}
# Cache for preprocessing configs per model (loaded from checkpoint)
_model_preprocessing_configs = {}


def get_device():
    """Get device (cuda if available, else cpu)"""
    if DEVICE == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_skeleton_model(model_type: str, device: Optional[torch.device] = None, 
                       prefer_anomaly: bool = False) -> torch.nn.Module:
    """
    Load a skeleton-based model from checkpoint.
    Models are cached in memory to avoid reloading.
    
    Args:
        model_type: Model type ('3dcnn_simple', '3dcnn_deep', '2dcnn', 'vit', 'stgcn', 'tcnt')
        device: Device to load model on
        prefer_anomaly: If True, prefer anomaly checkpoint if available
        
    Returns:
        Loaded model in eval mode
    """
    if device is None:
        device = get_device()
    
    # Check cache (use separate cache key for anomaly models)
    cache_key = f"{model_type}_{device}_{'anomaly' if prefer_anomaly else 'regular'}"
    if cache_key in _model_cache:
        return _model_cache[cache_key]

    # Supervised fall/no-fall models (2dcnn_resnet_fall, 3dcnn_simple_fall, vit_fall):
    # trained directly on binary Fall/No-Fall labels by evaluation/train_cnn_models.py.
    # These checkpoints have no 'config' key (just model_state_dict/model_type/num_classes),
    # so they're loaded separately from the KTH 6-activity checkpoints below.
    if model_type in FALL_DETECTION_CNN_CHECKPOINTS:
        checkpoint_path = FALL_DETECTION_CNN_CHECKPOINTS[model_type]
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint not found for {model_type}: {checkpoint_path}\n"
                f"Train one with: python evaluation/train_cnn_models.py"
            )

        print(f"Loading {model_type} model from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Architecture name is stored in the checkpoint itself (e.g. '2dcnn_resnet')
        base_model_type = checkpoint.get('model_type', model_type.replace('_fall', ''))
        num_classes = checkpoint.get('num_classes', len(FALL_DETECTION_CLASS_NAMES))

        _model_class_names[model_type] = FALL_DETECTION_CLASS_NAMES
        _model_preprocessing_configs[model_type] = {
            'clip_length': 32,
            'overlap': 0.5,
            'normalize': {'scale_to_01': False, 'center_on_hip': False},
            'image_width': 160,
            'image_height': 120,
        }

        model = create_model(base_model_type, num_classes=num_classes)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()

        _model_cache[cache_key] = model
        print(f"Model {model_type} loaded and cached (val_acc={checkpoint.get('val_acc', 'n/a')})")
        return model

    # Get checkpoint path (prefer anomaly checkpoint if requested)
    checkpoint_path = get_checkpoint_path(model_type, use_anomaly_detection=prefer_anomaly)
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found for {model_type}: {checkpoint_path}\n"
            f"Please ensure the model has been trained and checkpoint exists."
        )
    
    print(f"Loading {model_type} model from {checkpoint_path}...")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get config from checkpoint or use defaults
    if 'config' in checkpoint:
        config = checkpoint['config']
        num_classes = config['dataset']['num_classes']
        # Load class names from checkpoint (CRITICAL: must match training)
        if 'dataset' in config and 'actions' in config['dataset']:
            class_names = config['dataset']['actions']
            _model_class_names[model_type] = class_names
            print(f"Loaded class names from checkpoint: {class_names}")
        else:
            class_names = CLASS_NAMES  # Fallback
            print(f"Warning: No class names in checkpoint, using default: {class_names}")
        
        # Load preprocessing config from checkpoint (CRITICAL: must match training)
        if 'extraction' in config:
            extraction_config = config['extraction']
            preprocessing_config = {
                'clip_length': extraction_config.get('clip_length', 32),
                'overlap': extraction_config.get('overlap', 0.5),
                'normalize': extraction_config.get('normalize', {}),
                'image_width': extraction_config.get('image_width', 160),
                'image_height': extraction_config.get('image_height', 120),
            }
            _model_preprocessing_configs[model_type] = preprocessing_config
            print(f"Loaded preprocessing config from checkpoint: clip_length={preprocessing_config['clip_length']}, overlap={preprocessing_config['overlap']}")
        else:
            print(f"Warning: No extraction config in checkpoint, using defaults")
    else:
        # Try to load from base config
        try:
            config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs', 'base_config.yaml')
            config = load_config(config_path)
            num_classes = config['dataset']['num_classes']
            class_names = config['dataset'].get('actions', CLASS_NAMES)
            _model_class_names[model_type] = class_names
        except:
            num_classes = len(CLASS_NAMES)
            class_names = CLASS_NAMES
            print(f"Warning: Could not load config, using default class names: {class_names}")
    
    # Create model
    # Extract model-specific parameters from config (exclude type, skeleton_source, name, input_shape, layers)
    # Model config files contain documentation keys that shouldn't be passed to model constructor
    model_kwargs = {}
    if 'config' in checkpoint and 'model' in checkpoint['config']:
        model_config = checkpoint['config']['model']
        # Filter out non-model-constructor parameters (these are for documentation/reference only)
        excluded_keys = ['type', 'skeleton_source', 'name', 'input_shape', 'layers']
        model_kwargs = {k: v for k, v in model_config.items() 
                        if k not in excluded_keys and not isinstance(v, dict)}
    model = create_model(model_type, num_classes=num_classes, **model_kwargs)
    
    # Load weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Try direct state dict
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    # Cache model
    _model_cache[cache_key] = model
    print(f"Model {model_type} loaded and cached")
    
    return model


def load_vlm_model(device: Optional[torch.device] = None) -> Optional[VisionLanguageModel]:
    """
    Load VLM model (CLIP). Model is cached globally.
    Clears GPU cache before loading to avoid OOM errors.
    
    Args:
        device: Device to load model on
        
    Returns:
        VisionLanguageModel instance or None if not available
    """
    global _vlm_model
    
    if not VLM_AVAILABLE:
        return None
    
    if _vlm_model is not None:
        return _vlm_model
    
    if device is None:
        device = get_device()
    
    print("Loading VLM model (CLIP)...")
    try:
        # Clear GPU cache before loading to avoid OOM
        if device.type == 'cuda' and torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"GPU memory before VLM load: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
        
        # Load VLM config
        vlm_config_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 
            'configs', 
            'vlm_config.yaml'
        )
        if os.path.exists(vlm_config_path):
            vlm_config = load_vlm_config(vlm_config_path)
            model_name = vlm_config.get('model', {}).get('clip_model_name', 'openai/clip-vit-base-patch32')
        else:
            model_name = 'openai/clip-vit-base-patch32'
        
        _vlm_model = VisionLanguageModel(model_name=model_name, device=str(device))
        
        # Clear cache again after loading
        if device.type == 'cuda' and torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"GPU memory after VLM load: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
        
        print("VLM model loaded and cached")
        return _vlm_model
    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
            print(f"CUDA OOM error loading VLM model: {e}")
            # Try to clear cache and use CPU as fallback
            if device.type == 'cuda':
                torch.cuda.empty_cache()
                print("Attempting to load VLM on CPU as fallback...")
                try:
                    _vlm_model = VisionLanguageModel(model_name=model_name, device='cpu')
                    print("VLM model loaded on CPU")
                    return _vlm_model
                except Exception as e2:
                    print(f"Failed to load VLM on CPU: {e2}")
        print(f"Failed to load VLM model: {e}")
        return None
    except Exception as e:
        print(f"Failed to load VLM model: {e}")
        import traceback
        traceback.print_exc()
        return None


def get_class_names_for_model(model_type: str) -> List[str]:
    """
    Get class names for a model (from checkpoint if available, else default).
    CRITICAL: This ensures class indices match between training and inference.
    
    Args:
        model_type: Model type
        
    Returns:
        List of class names in the same order as training
    """
    # Check if we have cached class names for this model
    if model_type in _model_class_names:
        return _model_class_names[model_type]

    if model_type in FALL_DETECTION_CNN_CHECKPOINTS:
        return FALL_DETECTION_CLASS_NAMES

    # Try to load from checkpoint
    checkpoint_path = MODEL_CHECKPOINTS.get(model_type)
    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            if 'config' in checkpoint:
                config = checkpoint['config']
                if 'dataset' in config and 'actions' in config['dataset']:
                    class_names = config['dataset']['actions']
                    _model_class_names[model_type] = class_names
                    
                    # Verify against default (warn if mismatch)
                    if class_names != CLASS_NAMES:
                        print(f"⚠️  WARNING: Class names from checkpoint differ from default!")
                        print(f"   Checkpoint: {class_names}")
                        print(f"   Default:    {CLASS_NAMES}")
                        print(f"   Using checkpoint class names to ensure correct mapping.")
                    
                    return class_names
        except Exception as e:
            print(f"Warning: Could not load class names from checkpoint: {e}")
    
    # Fallback to default
    print(f"Using default class names for {model_type}: {CLASS_NAMES}")
    return CLASS_NAMES


def get_preprocessing_config_for_model(model_type: str) -> Dict[str, any]:
    """
    Get preprocessing configuration for a specific model (from checkpoint cache).
    
    This ensures that inference uses the same preprocessing settings as training.
    
    Args:
        model_type: Model type identifier
        
    Returns:
        Dictionary with preprocessing settings:
        - clip_length: Number of frames per clip
        - overlap: Overlap ratio (0.0 to 1.0)
        - normalize: Normalization settings dict
        - image_width: Image width for normalization
        - image_height: Image height for normalization
    """
    # Check if we have cached preprocessing config for this model
    if model_type in _model_preprocessing_configs:
        return _model_preprocessing_configs[model_type]
    
    # Try to load from checkpoint (try both regular and anomaly checkpoints)
    checkpoint_path = get_checkpoint_path(model_type, use_anomaly_detection=False)
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        # Try anomaly checkpoint as fallback
        checkpoint_path = ANOMALY_MODEL_CHECKPOINTS.get(model_type)
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            if 'config' in checkpoint:
                config = checkpoint['config']
                if 'extraction' in config:
                    extraction_config = config['extraction']
                    preprocessing_config = {
                        'clip_length': extraction_config.get('clip_length', 32),
                        'overlap': extraction_config.get('overlap', 0.5),
                        'normalize': extraction_config.get('normalize', {}),
                        'image_width': extraction_config.get('image_width', 160),
                        'image_height': extraction_config.get('image_height', 120),
                    }
                    _model_preprocessing_configs[model_type] = preprocessing_config
                    return preprocessing_config
        except Exception as e:
            print(f"Warning: Could not load preprocessing config from checkpoint: {e}")
    
    # Fallback to webapp config defaults
    from config import CLIP_LENGTH, OVERLAP, NORMALIZE_SCALE_TO_01, NORMALIZE_CENTER_ON_HIP, IMAGE_WIDTH, IMAGE_HEIGHT
    return {
        'clip_length': CLIP_LENGTH,
        'overlap': OVERLAP,
        'normalize': {
            'scale_to_01': NORMALIZE_SCALE_TO_01,
            'center_on_hip': NORMALIZE_CENTER_ON_HIP,
        },
        'image_width': IMAGE_WIDTH,
        'image_height': IMAGE_HEIGHT,
    }


def predict_skeleton_model(video_path: str, model_type: str, return_per_window: bool = False, 
                           use_anomaly_detection: bool = False) -> Dict:
    """
    Run inference on a video using a skeleton-based model.
    
    Args:
        video_path: Path to video file
        model_type: Model type ('3dcnn_simple', '3dcnn_deep', '2dcnn', 'vit', 'stgcn', 'tcnt')
        return_per_window: If True, return per-window probabilities and window info
        
    Returns:
        If return_per_window=False: Dictionary mapping class names to averaged probabilities
        If return_per_window=True: Dictionary with 'averaged' probabilities and 'per_window' list
    """
    device = get_device()
    
    # Load model (prefer anomaly checkpoint if anomaly detection is enabled)
    model = load_skeleton_model(model_type, device, prefer_anomaly=use_anomaly_detection)
    
    # Get class names for this model (from checkpoint, now cached)
    model_class_names = get_class_names_for_model(model_type)
    
    # Get preprocessing config from checkpoint (CRITICAL: must match training)
    preprocessing_config = get_preprocessing_config_for_model(model_type)
    clip_length = preprocessing_config['clip_length']
    overlap = preprocessing_config['overlap']
    normalize_config = preprocessing_config.get('normalize', {})
    
    print(f"Using preprocessing settings from checkpoint: clip_length={clip_length}, overlap={overlap}")
    
    # Extract and preprocess video
    from video_processor import extract_with_yolov11, normalize_keypoints as normalize_keypoints_webapp
    from preprocessing import generate_clips
    from config import YOLOV11_MODEL_NAME, YOLOV11_CONFIDENCE
    
    # Extract skeleton (now returns tuple: keypoints, bounding_boxes)
    extraction_result = extract_with_yolov11(
        video_path=video_path,
        model_name=YOLOV11_MODEL_NAME,
        confidence=YOLOV11_CONFIDENCE
    )
    
    # Handle both old (array) and new (tuple) return formats
    if isinstance(extraction_result, tuple):
        keypoints, bounding_boxes = extraction_result
    else:
        keypoints = extraction_result
        bounding_boxes = None
    
    if keypoints is None or (hasattr(keypoints, 'shape') and keypoints.shape[0] == 0):
        raise ValueError("Failed to extract skeleton from video")
    
    # Normalize keypoints based on checkpoint config (CRITICAL: must match training)
    # Check if normalization was used during training
    if normalize_config.get('scale_to_01', False) or normalize_config.get('center_on_hip', False):
        normalized_keypoints = normalize_keypoints_webapp(keypoints)
    else:
        # No normalization - use raw keypoints (matches training data format)
        normalized_keypoints = keypoints
    
    # Generate clips using settings from checkpoint
    clips = generate_clips(normalized_keypoints, clip_length=clip_length, overlap=overlap)
    
    if len(clips) == 0:
        raise ValueError("No clips generated from video")
    
    # Calculate window step (for overlap)
    step = int(clip_length * (1 - overlap))
    step = max(1, step)
    
    # Run inference on all clips
    all_probs = []
    window_info_list = []
    
    with torch.no_grad():
        for window_idx, clip in enumerate(clips):
            # Preprocess clip based on model type
            if model_type in ['3dcnn_simple', '3dcnn_deep', '3dcnn_simple_fall']:
                skeleton_tensor = torch.from_numpy(clip).float()
                skeleton_tensor = skeleton_tensor.permute(2, 1, 0)  # (C, J, T)
            elif model_type in ['2dcnn_resnet', '2dcnn_lenet', '2dcnn']:
                from preprocessing import create_2d_image_representation
                image_2d = create_2d_image_representation(clip)
                skeleton_tensor = torch.from_numpy(image_2d).float()
                skeleton_tensor = skeleton_tensor.unsqueeze(0)  # (1, T, 2J)
            elif model_type == 'vit':
                # ViT: [C=1, H=T, W=2J] (2D image like 2D CNN)
                from preprocessing import create_2d_image_representation
                image_2d = create_2d_image_representation(clip)
                skeleton_tensor = torch.from_numpy(image_2d).float()
                skeleton_tensor = skeleton_tensor.unsqueeze(0)  # (1, T, 2J)
            elif model_type in ['2dcnn_resnet_fall', 'vit_fall']:
                # Supervised fall/no-fall checkpoints were trained with a plain
                # interleaved reshape (see evaluation/train_cnn_models.py), NOT
                # create_2d_image_representation's x-block/y-block layout -- must
                # match exactly or the trained weights see a scrambled input.
                T, J, C = clip.shape
                image_2d = clip.reshape(T, J * C)
                skeleton_tensor = torch.from_numpy(image_2d).float()
                skeleton_tensor = skeleton_tensor.unsqueeze(0)  # (1, T, J*2)
            elif model_type in ['stgcn', 'tcnt']:
                skeleton_tensor = torch.from_numpy(clip).float()
                skeleton_tensor = skeleton_tensor.permute(2, 1, 0)  # (C, J, T)
            else:
                raise ValueError(f"Unknown model_type: {model_type}")
            
            # Add batch dimension
            clip_batch = skeleton_tensor.unsqueeze(0).to(device)
            
            # Forward pass
            outputs = model(clip_batch)
            probs = F.softmax(outputs, dim=1)
            probs_np = probs.cpu().numpy()[0]
            all_probs.append(probs_np)
            
            # Calculate window frame range
            window_start = window_idx * step
            window_end = window_start + clip_length - 1
            if window_end >= keypoints.shape[0]:
                window_end = keypoints.shape[0] - 1
            
            # Get class names for this model (from checkpoint)
            model_class_names = get_class_names_for_model(model_type)
            window_info_list.append({
                'window_num': window_idx,
                'window_start': window_start,
                'window_end': window_end,
                'probabilities': {model_class_names[i]: float(probs_np[i]) for i in range(len(model_class_names))}
            })
    
    # Average probabilities across all clips
    avg_probs = np.mean(all_probs, axis=0)
    
    # Get class names for this model (from checkpoint)
    model_class_names = get_class_names_for_model(model_type)
    
    # Create result dictionary
    result = {}
    for i, class_name in enumerate(model_class_names):
        if i < len(avg_probs):
            result[class_name] = float(avg_probs[i])
        else:
            print(f"Warning: Class index {i} out of range for probabilities (len={len(avg_probs)})")
    
    # Anomaly detection if enabled
    anomaly_result = None
    if use_anomaly_detection:
        # Get anomaly detection config from checkpoint
        # Try to get from already-loaded checkpoint (if we loaded anomaly checkpoint)
        # Otherwise load anomaly checkpoint to get threshold
        checkpoint_path = get_checkpoint_path(model_type, use_anomaly_detection=True)
        if checkpoint_path and os.path.exists(checkpoint_path):
            try:
                # Load checkpoint to get anomaly config (we need threshold)
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                config = checkpoint.get('config', {})
                anomaly_config = config.get('anomaly_detection', {})
                
                if anomaly_config.get('enabled', False):
                    threshold = anomaly_config.get('threshold')
                    method = anomaly_config.get('method', 'hybrid')
                    confidence_weight = anomaly_config.get('confidence_weight', 0.6)
                    entropy_weight = anomaly_config.get('entropy_weight', 0.4)
                    num_classes = len(model_class_names)
                    
                    if threshold is not None:
                        # Detect anomaly using averaged probabilities
                        anomaly_result = detect_anomaly(
                            avg_probs,
                            threshold=threshold,
                            method=method,
                            confidence_weight=confidence_weight,
                            entropy_weight=entropy_weight,
                            num_classes=num_classes
                        )
                        
                        # Also compute per-window anomaly detection
                        per_window_anomaly = []
                        for window_probs in all_probs:
                            window_anomaly = detect_anomaly(
                                window_probs,
                                threshold=threshold,
                                method=method,
                                confidence_weight=confidence_weight,
                                entropy_weight=entropy_weight,
                                num_classes=num_classes
                            )
                            per_window_anomaly.append({
                                'anomaly_score': float(window_anomaly['anomaly_score']),
                                'is_anomaly': bool(window_anomaly['is_anomaly']),
                                'confidence': float(window_anomaly['confidence']),
                                'entropy': float(window_anomaly['entropy'])
                            })
                        
                        anomaly_result['per_window'] = per_window_anomaly
                    else:
                        print("Warning: Anomaly detection threshold not found in checkpoint")
            except Exception as e:
                print(f"Warning: Could not load anomaly detection config: {e}")
    
    # Build result dictionary
    if return_per_window:
        result_dict = {
            'averaged': result,
            'per_window': window_info_list,
            'clip_length': clip_length,
            'overlap': overlap,
            'step': step,
            'total_frames': keypoints.shape[0],
            'keypoints': keypoints
        }
        if anomaly_result is not None:
            result_dict['anomaly_detection'] = anomaly_result
        return result_dict
    else:
        # Backward compatibility: return result directly when return_per_window=False
        if anomaly_result is not None:
            result['anomaly_detection'] = anomaly_result
        return result


def predict_vlm_fall_detection(video_path: str) -> Dict[str, any]:
    """
    Run VLM fall detection on a video.
    Uses the same approach as the command-line test script for consistent results.
    
    Args:
        video_path: Path to video file
        
    Returns:
        Dictionary with fall detection results
    """
    if not VLM_AVAILABLE or VisionLanguageModel is None:
        raise RuntimeError("VLM model is not available. Please ensure transformers library is installed and CLIP model can be loaded.")
    
    vlm = load_vlm_model()
    if vlm is None:
        raise RuntimeError("VLM model is not available. Please ensure transformers library is installed and CLIP model can be loaded.")
    
    # Load VLM config - this provides the correct prompts and parameters
    vlm_config = None
    vlm_config_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 
        'configs', 
        'vlm_config.yaml'
    )
    if os.path.exists(vlm_config_path) and load_vlm_config is not None:
        vlm_config = load_vlm_config(vlm_config_path)
    
    # Use zero-shot fall detection with config (same as test script)
    # This uses prompts from config file and correct temperature
    probabilities = zero_shot_fall_detection(
        video_path=video_path,
        vlm=vlm,
        config=vlm_config  # Pass config, let the function use prompts from config
    )
    
    # Get the fall probability (already calculated by zero_shot_fall_detection)
    fall_probability = probabilities.get("fall_probability", 0.0)
    
    # Use threshold from config (default 0.45 as per vlm_config.yaml)
    threshold = 0.45
    if vlm_config:
        threshold = vlm_config.get("few_shot", {}).get("threshold", 0.45)
    
    is_fall = fall_probability >= threshold
    
    # Return only aggregate result
    return {
        "fall_probability": float(fall_probability),
        "is_fall": bool(is_fall),
        "confidence": float(fall_probability),
        "threshold": float(threshold)
    }


def predict_vlm_description(video_path: str) -> Dict[str, str]:
    """
    Generate a text description of the video using VLM.
    Uses the same config settings as fall detection for consistent results.
    
    Args:
        video_path: Path to video file
        
    Returns:
        Dictionary with video description
    """
    if not VLM_AVAILABLE or VisionLanguageModel is None:
        raise RuntimeError("VLM model is not available. Please ensure transformers library is installed and CLIP model can be loaded.")
    
    vlm = load_vlm_model()
    if vlm is None:
        raise RuntimeError("VLM model is not available. Please ensure transformers library is installed and CLIP model can be loaded.")
    
    # Load VLM config - use SAME config as fall detection for consistent results
    vlm_config = None
    vlm_config_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 
        'configs', 
        'vlm_config.yaml'
    )
    if os.path.exists(vlm_config_path) and load_vlm_config is not None:
        vlm_config = load_vlm_config(vlm_config_path)
    
    # Use SAME prompts from config as fall detection for consistent results
    from vlm.config import get_zero_shot_prompts, get_num_frames, get_zero_shot_temperature, get_zero_shot_aggregation
    
    # Get parameters from config (same as fall detection)
    num_frames = 8
    temperature = 0.3
    aggregation = "mean"
    descriptive_prompts = None
    
    if vlm_config:
        num_frames = get_num_frames(vlm_config)
        temperature = get_zero_shot_temperature(vlm_config)
        aggregation = get_zero_shot_aggregation(vlm_config)
        descriptive_prompts = get_zero_shot_prompts(vlm_config)
    
    # Fallback prompts if config not available (same as vlm_config.yaml)
    if descriptive_prompts is None:
        descriptive_prompts = [
            # Fall detection prompts
            "a person falling down to the ground",
            "someone collapsing and falling",
            "a person losing balance and falling",
            "a person tripping and falling",
            "someone falling over",
            # Non-fall prompts
            "a person walking normally on the ground",
            "a person standing upright and stable",
            "a person sitting down normally",
            "a person moving around safely"
        ]
    
    # Load video frames - use the imported module
    import vlm.video_utils as video_utils_module
    get_clip_inputs_from_video = video_utils_module.get_clip_inputs_from_video
    
    video_frames = get_clip_inputs_from_video(
        video_path=video_path,
        num_frames=num_frames,
        processor=vlm.processor
    )
    
    # Get probabilities with same parameters as fall detection
    probabilities = vlm.get_text_probabilities_for_video(
        video_frames=video_frames,
        prompts=descriptive_prompts,
        aggregation=aggregation,
        temperature=temperature
    )
    
    # Identify fall-related prompts
    fall_prompts = [p for p in descriptive_prompts if "fall" in p.lower()]
    non_fall_prompts = [p for p in descriptive_prompts if "fall" not in p.lower()]
    
    # Calculate aggregate fall probability
    fall_probability = sum(probabilities.get(p, 0.0) for p in fall_prompts)
    
    # Get sorted probabilities
    sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    
    # Get top non-fall activity for context
    top_non_fall = [(p, prob) for p, prob in sorted_probs if "fall" not in p.lower()]
    top_fall = [(p, prob) for p, prob in sorted_probs if "fall" in p.lower()]
    
    # Create description based on fall probability
    if fall_probability >= 0.45:  # Same threshold as fall detection
        # Fall detected
        if top_fall:
            top_fall_desc = top_fall[0][0]
            description = f"⚠️ FALL DETECTED: The video shows {top_fall_desc}."
        else:
            description = "⚠️ FALL DETECTED: A person appears to be falling in this video."
        
        # Add confidence
        description += f" (Fall probability: {fall_probability*100:.1f}%)"
    else:
        # No fall - describe normal activity
        if top_non_fall and top_non_fall[0][1] > 0.1:
            main_activity = top_non_fall[0][0]
            description = f"The video shows {main_activity}."
            if len(top_non_fall) > 1 and top_non_fall[1][1] > 0.08:
                description += f" It may also show {top_non_fall[1][0]}."
        else:
            description = "The video shows normal human activity."
        
        # Add fall probability context
        description += f" (Fall probability: {fall_probability*100:.1f}%)"
    
    # Get top activities for display
    top_activities = [desc for desc, prob in sorted_probs[:5]]
    
    return {
        "description": description,
        "fall_probability": float(fall_probability),
        "is_fall": fall_probability >= 0.45,
        "top_activities": top_activities,
        "probabilities": {desc: float(prob) for desc, prob in sorted_probs[:5]}
    }


def predict(video_path: str, model_type: str, use_anomaly_detection: bool = False) -> Dict[str, float]:
    """
    Run inference on a video using the specified skeleton-based model.
    Note: VLM models are handled separately via /vlm/<mode> endpoint.
    
    Args:
        video_path: Path to video file
        model_type: Model type (3dcnn_simple, 3dcnn_deep, 2dcnn, vit, stgcn, tcnt)
        use_anomaly_detection: Whether to use anomaly detection mode
        
    Returns:
        Dictionary mapping class names to probabilities (and anomaly detection results if enabled)
    """
    return predict_skeleton_model(video_path, model_type, use_anomaly_detection=use_anomaly_detection)

