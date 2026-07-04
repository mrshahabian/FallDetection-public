"""
Evaluators for different fall detection methods.
Each evaluator processes videos in windows and measures inference time.
"""

import os
import sys
import time
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)


class BaseEvaluator:
    """Base class for all evaluators."""
    
    def __init__(self, window_length: int = 32, overlap: float = 0.5):
        """
        Initialize base evaluator.
        
        Args:
            window_length: Length of each time window in frames
            overlap: Overlap ratio between windows (0.0 to 1.0)
        """
        self.window_length = window_length
        self.overlap = overlap
        self.step = max(1, int(window_length * (1 - overlap)))
    
    def evaluate_video(self, video_path: str) -> Dict[str, Any]:
        """
        Evaluate a single video.
        
        Args:
            video_path: Path to video file
        
        Returns:
            Dictionary with:
            - 'is_fall': bool, whether fall detected (averaged)
            - 'fall_probability': float, probability of fall (0-1)
            - 'per_window': List of per-window results
            - 'inference_times': List of inference times per window
            - 'avg_inference_time': float, average inference time per window
        """
        raise NotImplementedError
    
    def evaluate_batch(self, video_paths: List[str]) -> Dict[str, Any]:
        """
        Evaluate a batch of videos.
        
        Args:
            video_paths: List of video file paths
        
        Returns:
            Dictionary with aggregated results across all videos
        """
        all_results = []
        all_inference_times = []
        
        for video_path in video_paths:
            try:
                result = self.evaluate_video(video_path)
                all_results.append(result)
                all_inference_times.extend(result.get('inference_times', []))
            except Exception as e:
                logger.error(f"Error evaluating {video_path}: {e}")
                continue
        
        if len(all_results) == 0:
            return {
                'num_videos': 0,
                'avg_fall_probability': 0.0,
                'fall_detection_rate': 0.0,
                'avg_inference_time': 0.0,
                'total_inference_time': 0.0
            }
        
        # Aggregate results
        avg_fall_prob = np.mean([r['fall_probability'] for r in all_results])
        fall_detection_rate = np.mean([r['is_fall'] for r in all_results])
        avg_inference_time = np.mean(all_inference_times) if all_inference_times else 0.0
        total_inference_time = np.sum(all_inference_times) if all_inference_times else 0.0
        
        return {
            'num_videos': len(all_results),
            'avg_fall_probability': float(avg_fall_prob),
            'fall_detection_rate': float(fall_detection_rate),
            'avg_inference_time': float(avg_inference_time),
            'total_inference_time': float(total_inference_time),
            'per_video_results': all_results
        }


class SimpleFallEvaluator(BaseEvaluator):
    """Evaluator for simple fall detection based on skeleton pose."""
    
    def __init__(self, window_length: int = 32, overlap: float = 0.5):
        super().__init__(window_length, overlap)
        # Import simple fall detection
        sys.path.insert(0, str(project_root / "webapp"))
        from simple_fall_detection import predict_simple_fall_detection
        self.predict_fn = predict_simple_fall_detection
    
    def evaluate_video(self, video_path: str) -> Dict[str, Any]:
        """Evaluate video using simple fall detection."""
        start_time = time.time()
        
        try:
            result = self.predict_fn(
                video_path=video_path,
                clip_length=self.window_length,
                overlap=self.overlap,
                return_per_window=True
            )
            
            inference_time = time.time() - start_time
            
            # Extract per-window results
            per_window = result.get('per_window', [])
            window_times = [inference_time / len(per_window)] * len(per_window) if per_window else []
            
            return {
                'is_fall': result.get('is_fall', False),
                'fall_probability': result.get('fall_probability', 0.0),
                'per_window': per_window,
                'inference_times': window_times,
                'avg_inference_time': inference_time / len(per_window) if per_window else inference_time,
                'total_inference_time': inference_time
            }
        except Exception as e:
            logger.error(f"Error in simple fall detection for {video_path}: {e}")
            return {
                'is_fall': False,
                'fall_probability': 0.0,
                'per_window': [],
                'inference_times': [],
                'avg_inference_time': 0.0,
                'total_inference_time': 0.0,
                'error': str(e)
            }


class ZeroShotVLMEvaluator(BaseEvaluator):
    """Evaluator for zero-shot VLM fall detection."""
    
    def __init__(self, window_length: int = 32, overlap: float = 0.5, 
                 num_frames: int = 1, device: str = "auto"):
        """
        Initialize zero-shot VLM evaluator.
        
        Args:
            window_length: Length of each time window in frames
            overlap: Overlap ratio between windows
            num_frames: Number of frames to sample per window (default: 1 for efficiency)
            device: Device to run inference on
        """
        super().__init__(window_length, overlap)
        self.num_frames = num_frames
        
        # Import VLM modules
        from vlm.vlm_model import VisionLanguageModel
        from vlm.zero_shot_inference import zero_shot_fall_detection, get_fall_detection_result
        from vlm.config import load_config
        
        # Load config
        config_path = project_root / "configs" / "vlm_config.yaml"
        if config_path.exists():
            self.config = load_config(str(config_path))
        else:
            self.config = {}
        
        # Initialize VLM
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        model_name = self.config.get("model", {}).get("clip_model_name", "openai/clip-vit-base-patch32")
        logger.info(f"Loading VLM model: {model_name} on {device}")
        self.vlm = VisionLanguageModel(model_name=model_name, device=device)

        self.zero_shot_fn = zero_shot_fall_detection
        self.get_result_fn = get_fall_detection_result

        # Encode the balanced fall/non-fall prompt bank once (not per-window): the
        # prompts never change between windows, so re-encoding them ~198 times per
        # video was pure overhead that inflated the measured per-window latency.
        from evaluation.prompts import get_all_prompts, PROMPT_INFO
        prompts = get_all_prompts()
        num_fall = PROMPT_INFO['num_fall_prompts']
        logger.info(f"Using {len(prompts)} balanced prompts ({num_fall} fall, {len(prompts) - num_fall} non-fall)")
        self.fall_text_embeds = self.vlm.encode_texts(prompts[:num_fall], normalize=True)
        self.non_fall_text_embeds = self.vlm.encode_texts(prompts[num_fall:], normalize=True)

    def evaluate_video(self, video_path: str) -> Dict[str, Any]:
        """Evaluate video using zero-shot VLM."""
        import cv2
        from vlm.video_utils import get_clip_inputs_from_video
        
        # Load video to get frame count
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        if total_frames == 0:
            return {
                'is_fall': False,
                'fall_probability': 0.0,
                'per_window': [],
                'inference_times': [],
                'avg_inference_time': 0.0,
                'total_inference_time': 0.0,
                'error': 'No frames in video'
            }
        
        # Generate window boundaries
        windows = []
        start = 0
        while start < total_frames:
            end = min(start + self.window_length, total_frames)
            windows.append((start, end))
            start += self.step
        
        if len(windows) == 0:
            windows = [(0, min(self.window_length, total_frames))]
        
        per_window = []
        inference_times = []
        
        for window_start, window_end in windows:
            try:
                # Create temporary video clip for this window
                # For efficiency, we'll sample frames directly from the video
                window_start_time = time.time()
                
                # Extract frames from the window range
                frames = self._extract_frames_from_range(video_path, window_start, window_end, self.num_frames)
                
                if len(frames) == 0:
                    continue
                
                # Preprocess frames for CLIP
                from PIL import Image
                pil_images = [Image.fromarray(frame) for frame in frames]
                inputs = self.vlm.processor(images=pil_images, return_tensors="pt")
                video_frames = inputs["pixel_values"]

                # CONTRASTIVE APPROACH: compare fall vs non-fall directly using the
                # prompt bank encoded once in __init__ (not re-encoded per window).
                # This gives much better separation than softmax over 198 prompts.
                image_embeds = self.vlm.encode_images(video_frames, normalize=True)
                fall_similarities = self.vlm.compute_similarity(image_embeds, self.fall_text_embeds, temperature=1.0)
                non_fall_similarities = self.vlm.compute_similarity(image_embeds, self.non_fall_text_embeds, temperature=1.0)
                
                # Aggregate over frames and prompts
                # Average similarity for fall prompts vs non-fall prompts
                fall_sim = fall_similarities.mean(dim=0).mean().item()  # Average over all fall prompts and frames
                non_fall_sim = non_fall_similarities.mean(dim=0).mean().item()  # Average over all non-fall prompts and frames
                
                # Contrastive probability: Use sigmoid of difference
                # This gives better separation than softmax over 198 prompts
                diff = fall_sim - non_fall_sim
                # Use temperature scaling for sigmoid (lower = sharper)
                temperature = 0.1  # Lower temperature for sharper probabilities
                fall_prob = torch.sigmoid(torch.tensor(diff / temperature)).item()

                window_time = time.time() - window_start_time
                inference_times.append(window_time)
                
                per_window.append({
                    'window_start': window_start,
                    'window_end': window_end,
                    'fall_probability': float(fall_prob),
                    'is_fall': fall_prob >= 0.5
                })
                
            except Exception as e:
                logger.error(f"Error processing window {window_start}-{window_end}: {e}")
                continue
        
        if len(per_window) == 0:
            return {
                'is_fall': False,
                'fall_probability': 0.0,
                'per_window': [],
                'inference_times': [],
                'avg_inference_time': 0.0,
                'total_inference_time': 0.0,
                'error': 'No windows processed'
            }
        
        # Aggregate results using majority vote across windows
        # Video is considered fall if majority of windows detect fall
        from evaluation.metrics import aggregate_window_results
        aggregated = aggregate_window_results(per_window, true_label=True)
        is_fall = aggregated['is_fall']
        avg_fall_prob = aggregated['fall_probability']
        
        return {
            'is_fall': is_fall,
            'fall_probability': float(avg_fall_prob),
            'per_window': per_window,
            'inference_times': inference_times,
            'avg_inference_time': float(np.mean(inference_times)) if inference_times else 0.0,
            'total_inference_time': float(np.sum(inference_times))
        }
    
    def _extract_frames_from_range(self, video_path: str, start_frame: int, end_frame: int, num_frames: int):
        """Extract frames from a specific range in the video."""
        import cv2
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        if end_frame <= start_frame:
            cap.release()
            return frames
        
        # Sample frame indices uniformly from the range
        if num_frames == 1:
            # Use middle frame
            frame_idx = (start_frame + end_frame) // 2
            frame_indices = [frame_idx]
        else:
            frame_indices = np.linspace(start_frame, end_frame - 1, num_frames, dtype=int)
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
        
        cap.release()
        return frames
    
    @staticmethod
    def _extract_frames_from_range(video_path: str, start_frame: int, end_frame: int, num_frames: int):
        """Extract frames from a specific range in the video."""
        import cv2
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        if end_frame <= start_frame:
            cap.release()
            return frames
        
        # Sample frame indices uniformly from the range
        if num_frames == 1:
            # Use middle frame
            frame_idx = (start_frame + end_frame) // 2
            frame_indices = [frame_idx]
        else:
            frame_indices = np.linspace(start_frame, end_frame - 1, num_frames, dtype=int)
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
        
        cap.release()
        return frames


class FewShotVLMEvaluator(BaseEvaluator):
    """Evaluator for few-shot VLM fall detection."""
    
    def __init__(self, classifier_path: str, window_length: int = 32, 
                 overlap: float = 0.5, num_frames: int = 1, device: str = "auto"):
        """
        Initialize few-shot VLM evaluator.
        
        Args:
            classifier_path: Path to trained classifier
            window_length: Length of each time window in frames
            overlap: Overlap ratio between windows
            num_frames: Number of frames to sample per window
            device: Device to run inference on
        """
        super().__init__(window_length, overlap)
        self.num_frames = num_frames
        self.classifier_path = classifier_path
        
        # Import VLM modules
        from vlm.vlm_model import VisionLanguageModel
        from vlm.few_shot_train import load_classifier, few_shot_predict
        from vlm.config import load_config
        
        # Load config
        config_path = project_root / "configs" / "vlm_config.yaml"
        if config_path.exists():
            self.config = load_config(str(config_path))
        else:
            self.config = {}
        
        # Initialize VLM
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        model_name = self.config.get("model", {}).get("clip_model_name", "openai/clip-vit-base-patch32")
        logger.info(f"Loading VLM model: {model_name} on {device}")
        self.vlm = VisionLanguageModel(model_name=model_name, device=device)
        
        # Load classifier
        embedding_dim = self.vlm.image_embed_dim
        logger.info(f"Loading classifier from {classifier_path}")
        self.classifier = load_classifier(classifier_path, input_dim=embedding_dim, hidden_dim=None)
        self.classifier = self.classifier.to(device)
        
        # Helper method for extracting frames (static method)
    
    def evaluate_video(self, video_path: str) -> Dict[str, Any]:
        """Evaluate video using few-shot VLM."""
        import cv2
        
        # Load video to get frame count
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        if total_frames == 0:
            return {
                'is_fall': False,
                'fall_probability': 0.0,
                'per_window': [],
                'inference_times': [],
                'avg_inference_time': 0.0,
                'total_inference_time': 0.0,
                'error': 'No frames in video'
            }
        
        # Generate window boundaries
        windows = []
        start = 0
        while start < total_frames:
            end = min(start + self.window_length, total_frames)
            windows.append((start, end))
            start += self.step
        
        if len(windows) == 0:
            windows = [(0, min(self.window_length, total_frames))]
        
        per_window = []
        inference_times = []
        
        for window_start, window_end in windows:
            try:
                window_start_time = time.time()
                
                # Extract frames from the window range
                frames = ZeroShotVLMEvaluator._extract_frames_from_range(video_path, window_start, window_end, self.num_frames)
                
                if len(frames) == 0:
                    continue
                
                # Preprocess frames for CLIP
                from PIL import Image
                pil_images = [Image.fromarray(frame) for frame in frames]
                inputs = self.vlm.processor(images=pil_images, return_tensors="pt")
                video_frames = inputs["pixel_values"].to(self.vlm.device)
                
                # Extract video embedding from frames
                frame_embeddings = self.vlm.encode_images(video_frames, normalize=True)
                video_embedding = frame_embeddings.mean(dim=0)  # Average over frames
                
                # Get prediction from classifier
                self.classifier.eval()
                with torch.no_grad():
                    logit = self.classifier(video_embedding.unsqueeze(0)).squeeze(-1)
                    probability = torch.sigmoid(logit).item()
                
                is_fall = probability >= 0.5
                
                window_time = time.time() - window_start_time
                inference_times.append(window_time)
                
                per_window.append({
                    'window_start': window_start,
                    'window_end': window_end,
                    'fall_probability': float(probability),
                    'is_fall': is_fall
                })
                
            except Exception as e:
                logger.error(f"Error processing window {window_start}-{window_end}: {e}")
                continue
        
        if len(per_window) == 0:
            return {
                'is_fall': False,
                'fall_probability': 0.0,
                'per_window': [],
                'inference_times': [],
                'avg_inference_time': 0.0,
                'total_inference_time': 0.0,
                'error': 'No windows processed'
            }
        
        # Aggregate results using majority vote across windows
        # Video is considered fall if majority of windows detect fall
        from evaluation.metrics import aggregate_window_results
        aggregated = aggregate_window_results(per_window, true_label=True)
        is_fall = aggregated['is_fall']
        avg_fall_prob = aggregated['fall_probability']
        
        return {
            'is_fall': is_fall,
            'fall_probability': float(avg_fall_prob),
            'per_window': per_window,
            'inference_times': inference_times,
            'avg_inference_time': float(np.mean(inference_times)) if inference_times else 0.0,
            'total_inference_time': float(np.sum(inference_times))
        }


class CNNEvaluator(BaseEvaluator):
    """Evaluator for CNN-based fall detection (2D CNN, 3D CNN, ViT)."""
    
    def __init__(self, checkpoint_path: str, model_type: str, 
                 window_length: int = 32, overlap: float = 0.5, device: str = "auto"):
        """
        Initialize CNN evaluator.
        
        Args:
            checkpoint_path: Path to model checkpoint
            model_type: Type of model ('2dcnn_resnet', '2dcnn_lenet', '3dcnn_simple', '3dcnn_deep', 'vit')
            window_length: Length of each time window in frames
            overlap: Overlap ratio between windows
            device: Device to run inference on
        """
        super().__init__(window_length, overlap)
        self.model_type = model_type
        self.checkpoint_path = checkpoint_path
        
        # Import model and utilities
        from src.models import create_model
        from src.utils import get_device as get_device_util
        
        # Get device
        if device == "auto":
            device = get_device_util()
        self.device = torch.device(device)
        
        # Load checkpoint
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Get model config from checkpoint or use defaults
        if 'config' in checkpoint:
            config = checkpoint['config']
            num_classes = config.get('dataset', {}).get('num_classes', 2)  # Binary: fall vs non-fall
        else:
            num_classes = 2
        
        # Create model
        logger.info(f"Creating {model_type} model with {num_classes} classes")
        self.model = create_model(model_type, num_classes=num_classes)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Import skeleton extraction
        from src.skeleton_extraction import extract_with_yolov11
        
        self.extract_skeletons = extract_with_yolov11
    
    def evaluate_video(self, video_path: str) -> Dict[str, Any]:
        """Evaluate video using CNN model."""
        try:
            # Extract skeletons from video (include this in total inference time)
            skeleton_extraction_start = time.time()
            result = self.extract_skeletons(video_path)
            if isinstance(result, tuple):
                skeletons, _ = result  # Extract keypoints, ignore bboxes
            else:
                skeletons = result
            skeleton_extraction_time = time.time() - skeleton_extraction_start
            
            if skeletons is None or len(skeletons) == 0:
                return {
                    'is_fall': False,
                    'fall_probability': 0.0,
                    'per_window': [],
                    'inference_times': [],
                    'avg_inference_time': 0.0,
                    'total_inference_time': skeleton_extraction_time,
                    'skeleton_extraction_time': skeleton_extraction_time,
                    'error': 'No skeletons extracted'
                }
            
            # Generate windows
            total_frames = len(skeletons)
            windows = []
            start = 0
            while start < total_frames:
                end = min(start + self.window_length, total_frames)
                windows.append((start, end))
                start += self.step
            
            if len(windows) == 0:
                windows = [(0, min(self.window_length, total_frames))]
            
            per_window = []
            inference_times = []
            
            for window_start, window_end in windows:
                try:
                    window_start_time = time.time()
                    
                    # Extract window skeleton
                    window_skeleton = skeletons[window_start:window_end]
                    
                    if len(window_skeleton) < self.window_length:
                        # Pad if necessary
                        padding = np.zeros((self.window_length - len(window_skeleton), 
                                          window_skeleton.shape[1], 
                                          window_skeleton.shape[2]))
                        window_skeleton = np.concatenate([window_skeleton, padding], axis=0)
                    
                    # Convert to tensor and prepare for model
                    if self.model_type in ['3dcnn_simple', '3dcnn_deep']:
                        # Input skeleton is [T, J, C] where T=frames, J=joints, C=coordinates
                        # Model expects [B, C, J, T] where B=batch, C=2, J=17, T=32
                        # Transpose: [T, J, C] -> [C, J, T]
                        skeleton_tensor = torch.from_numpy(window_skeleton).float()  # [T, J, C]
                        skeleton_tensor = skeleton_tensor.permute(2, 1, 0)  # [C, J, T]
                        skeleton_tensor = skeleton_tensor.unsqueeze(0)  # [B, C, J, T]
                    elif self.model_type in ['2dcnn_resnet', '2dcnn_lenet', 'vit']:
                        # Reshape to 2D image: [1, 32, 34] -> [1, 1, 32, 34]
                        # Assuming skeleton is [T, J*2] -> reshape to [1, T, J*2]
                        if window_skeleton.ndim == 3:
                            # [T, J, 2] -> [T, J*2]
                            T, J, C = window_skeleton.shape
                            window_skeleton = window_skeleton.reshape(T, J * C)
                        skeleton_tensor = torch.from_numpy(window_skeleton).float()
                        skeleton_tensor = skeleton_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, T, J*2]
                    else:
                        skeleton_tensor = torch.from_numpy(window_skeleton).float()
                        if skeleton_tensor.dim() == 3:
                            skeleton_tensor = skeleton_tensor.unsqueeze(0)
                    
                    skeleton_tensor = skeleton_tensor.to(self.device)
                    
                    # Run inference
                    with torch.no_grad():
                        outputs = self.model(skeleton_tensor)
                        probs = torch.softmax(outputs, dim=1)
                        
                        # For binary classification, fall is class 1
                        # Check if model outputs are reversed (class 0 = fall, class 1 = non-fall)
                        # This can happen if model was trained with reversed labels
                        if probs.shape[1] == 2:
                            prob_class_0 = probs[0, 0].item()
                            prob_class_1 = probs[0, 1].item()
                            
                            # Use class 1 as fall (standard mapping: 0=non-fall, 1=fall)
                            fall_prob = prob_class_1
                            
                            # DIAGNOSTIC: If model seems reversed (class 0 has higher prob for fall videos),
                            # we might need to invert. For now, use class 1 as fall.
                            # If results show reversed behavior, we'll need to retrain or invert.
                        else:
                            # Single class output (sigmoid)
                            fall_prob = torch.sigmoid(outputs[0, 0]).item() if outputs.shape[1] == 1 else probs[0, 0].item()
                        
                        is_fall = fall_prob >= 0.5
                    
                    window_time = time.time() - window_start_time
                    inference_times.append(window_time)
                    
                    per_window.append({
                        'window_start': window_start,
                        'window_end': window_end,
                        'fall_probability': float(fall_prob),
                        'is_fall': is_fall
                    })
                    
                except Exception as e:
                    logger.error(f"Error processing window {window_start}-{window_end}: {e}")
                    continue
            
            if len(per_window) == 0:
                return {
                    'is_fall': False,
                    'fall_probability': 0.0,
                    'per_window': [],
                    'inference_times': [],
                    'avg_inference_time': 0.0,
                    'total_inference_time': 0.0,
                    'error': 'No windows processed'
                }
            
            # Aggregate results using majority vote across windows
            # Video is considered fall if majority of windows detect fall
            from evaluation.metrics import aggregate_window_results
            aggregated = aggregate_window_results(per_window, true_label=True)
            is_fall = aggregated['is_fall']
            avg_fall_prob = aggregated['fall_probability']
            
            # Calculate total inference time including skeleton extraction
            # Distribute skeleton extraction time across windows
            total_model_inference = float(np.sum(inference_times))
            skeleton_time_per_window = skeleton_extraction_time / len(windows) if len(windows) > 0 else 0.0
            total_inference_time = skeleton_extraction_time + total_model_inference
            avg_inference_time = (total_inference_time / len(windows)) if len(windows) > 0 else 0.0
            
            # Update per-window inference times to include skeleton extraction portion
            adjusted_inference_times = [t + skeleton_time_per_window for t in inference_times]
            
            return {
                'is_fall': is_fall,
                'fall_probability': float(avg_fall_prob),
                'per_window': per_window,
                'inference_times': adjusted_inference_times,
                'avg_inference_time': avg_inference_time,
                'total_inference_time': total_inference_time,
                'skeleton_extraction_time': skeleton_extraction_time,
                'model_inference_time': total_model_inference
            }
            
        except Exception as e:
            logger.error(f"Error in CNN evaluation for {video_path}: {e}")
            return {
                'is_fall': False,
                'fall_probability': 0.0,
                'per_window': [],
                'inference_times': [],
                'avg_inference_time': 0.0,
                'total_inference_time': 0.0,
                'error': str(e)
            }

