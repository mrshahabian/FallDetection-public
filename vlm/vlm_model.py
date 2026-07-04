"""
Vision-Language Model (VLM) wrapper for CLIP.

This module provides a clean interface to HuggingFace CLIP models for
encoding images and text, and computing similarities between them.
"""

import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor
from typing import List, Dict, Optional, Union
import logging

logger = logging.getLogger(__name__)


class VisionLanguageModel:
    """
    Wrapper class for CLIP model from HuggingFace Transformers.
    
    This class encapsulates the CLIP model and provides convenient methods
    for encoding images and text, and computing similarities between them.
    
    Example:
        >>> vlm = VisionLanguageModel("openai/clip-vit-base-patch32", device="cuda")
        >>> image_embeds = vlm.encode_images(image_tensor)
        >>> text_embeds = vlm.encode_texts(["a person falling", "a person standing"])
        >>> similarities = vlm.compute_similarity(image_embeds, text_embeds)
    """
    
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: Union[str, torch.device] = "auto"
    ):
        """
        Initialize the Vision-Language Model.
        
        Args:
            model_name: HuggingFace model identifier (e.g., "openai/clip-vit-base-patch32").
            device: Device to run model on ("cuda", "cpu", or "auto" for auto-detection).
        """
        # Auto-detect device if needed
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = torch.device(device)
        self.model_name = model_name
        
        logger.info(f"Loading CLIP model: {model_name}")
        logger.info(f"Using device: {self.device}")
        
        # Load CLIP model and processor
        try:
            self.model = CLIPModel.from_pretrained(model_name).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(model_name)
            self.model.eval()  # Set to evaluation mode
            logger.info("CLIP model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
            raise
        
        # Get embedding dimensions
        self.image_embed_dim = self.model.config.projection_dim
        self.text_embed_dim = self.model.config.projection_dim
        
        logger.info(f"Image embedding dimension: {self.image_embed_dim}")
        logger.info(f"Text embedding dimension: {self.text_embed_dim}")
    
    def encode_images(
        self,
        images: torch.Tensor,
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Encode images into normalized embeddings.
        
        Args:
            images: Tensor of shape [N, 3, H, W] or [N, C, H, W] with preprocessed images.
                   Images should be normalized according to CLIP preprocessing.
            normalize: Whether to L2-normalize embeddings (default: True).
        
        Returns:
            Tensor of shape [N, D] with image embeddings (normalized if normalize=True).
        """
        with torch.no_grad():
            # Move to device if needed
            if images.device != self.device:
                images = images.to(self.device)
            
            # Get image features from CLIP vision encoder. Pinned to transformers<5.0.0,
            # which returns a plain tensor here; transformers>=5.0 returns a
            # BaseModelOutputWithPooling instead, with the embedding in .pooler_output.
            image_outputs = self.model.get_image_features(pixel_values=images)
            image_outputs = getattr(image_outputs, "pooler_output", image_outputs)

            # Normalize embeddings to unit norm (for cosine similarity)
            if normalize:
                image_outputs = F.normalize(image_outputs, p=2, dim=-1)

        return image_outputs
    
    def encode_texts(
        self,
        texts: Union[str, List[str]],
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Encode text prompts into normalized embeddings.
        
        Args:
            texts: Single string or list of strings to encode.
            normalize: Whether to L2-normalize embeddings (default: True).
        
        Returns:
            Tensor of shape [T, D] with text embeddings (normalized if normalize=True),
            where T is the number of texts.
        """
        # Convert single string to list
        if isinstance(texts, str):
            texts = [texts]
        
        with torch.no_grad():
            # Tokenize and encode texts
            inputs = self.processor(
                text=texts,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.device)
            
            # Get text features from CLIP text encoder (see encode_images for why
            # we unwrap .pooler_output: transformers>=5.0 returns a ModelOutput here).
            text_outputs = self.model.get_text_features(**inputs)
            text_outputs = getattr(text_outputs, "pooler_output", text_outputs)

            # Normalize embeddings to unit norm (for cosine similarity)
            if normalize:
                text_outputs = F.normalize(text_outputs, p=2, dim=-1)
        
        return text_outputs
    
    def compute_similarity(
        self,
        image_embeds: torch.Tensor,
        text_embeds: torch.Tensor,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Compute cosine similarity between image and text embeddings.
        
        Args:
            image_embeds: Tensor of shape [N, D] with image embeddings.
            text_embeds: Tensor of shape [T, D] with text embeddings.
            temperature: Temperature scaling for similarity scores (default: 1.0).
        
        Returns:
            Tensor of shape [N, T] with cosine similarities (scaled by temperature).
        """
        # Ensure embeddings are on the same device
        if image_embeds.device != text_embeds.device:
            text_embeds = text_embeds.to(image_embeds.device)
        
        # Compute cosine similarity: (N, D) @ (D, T) -> (N, T)
        # Since embeddings are normalized, dot product = cosine similarity
        similarities = torch.matmul(image_embeds, text_embeds.t())
        
        # Apply temperature scaling
        if temperature != 1.0:
            similarities = similarities / temperature
        
        return similarities
    
    def get_text_probabilities_for_video(
        self,
        video_frames: torch.Tensor,
        prompts: List[str],
        aggregation: str = "mean",
        temperature: float = 1.0
    ) -> Dict[str, float]:
        """
        Get probability distribution over text prompts for a video.
        
        This is a convenience method that combines encoding and similarity computation
        to get probabilities for each prompt.
        
        Args:
            video_frames: Tensor of shape [N, 3, H, W] with preprocessed video frames.
            prompts: List of text prompts to evaluate.
            aggregation: How to aggregate frame-level similarities: "mean" or "max".
            temperature: Temperature for softmax (default: 1.0).
        
        Returns:
            Dictionary mapping each prompt to its probability.
        """
        # Encode frames and prompts
        image_embeds = self.encode_images(video_frames, normalize=True)
        text_embeds = self.encode_texts(prompts, normalize=True)
        
        # Compute similarities: [N, T]
        similarities = self.compute_similarity(image_embeds, text_embeds, temperature=1.0)
        
        # Aggregate over frames
        if aggregation == "mean":
            video_similarity = similarities.mean(dim=0)  # [T]
        elif aggregation == "max":
            video_similarity = similarities.max(dim=0)[0]  # [T]
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation}")
        
        # Apply temperature scaling for softmax
        if temperature != 1.0:
            video_similarity = video_similarity / temperature
        
        # Convert to probabilities using softmax
        probabilities = F.softmax(video_similarity, dim=0)
        
        # Create dictionary mapping prompts to probabilities
        result = {
            prompt: prob.item()
            for prompt, prob in zip(prompts, probabilities)
        }
        
        return result
    
    def __repr__(self) -> str:
        """String representation of the model."""
        return (
            f"VisionLanguageModel("
            f"model_name='{self.model_name}', "
            f"device={self.device}, "
            f"embed_dim={self.image_embed_dim}"
            f")"
        )











