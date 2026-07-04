"""
Few-shot training for fall detection using linear probing on CLIP embeddings.

This module implements few-shot adaptation by training a small classifier
on top of frozen CLIP image embeddings. The CLIP model itself is not fine-tuned,
only a lightweight linear classifier is trained.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Tuple, Dict, Optional, Any
import logging
from tqdm import tqdm
import os

from .vlm_model import VisionLanguageModel
from .video_utils import get_clip_inputs_from_video
from .config import get_num_frames, get_few_shot_config

logger = logging.getLogger(__name__)


class VideoEmbeddingDataset(Dataset):
    """
    Dataset for video embeddings and labels.
    
    This dataset stores pre-computed video embeddings (from CLIP) along with
    their labels for training a classifier.
    """
    
    def __init__(self, embeddings: torch.Tensor, labels: torch.Tensor):
        """
        Initialize dataset.
        
        Args:
            embeddings: Tensor of shape [N, D] with video embeddings.
            labels: Tensor of shape [N] with labels (0 or 1 for binary classification).
        """
        self.embeddings = embeddings
        self.labels = labels
    
    def __len__(self) -> int:
        return len(self.embeddings)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.embeddings[idx], self.labels[idx]


class LinearClassifier(nn.Module):
    """
    Simple linear classifier for binary classification.
    
    This is a lightweight classifier that takes CLIP embeddings as input
    and outputs a binary classification (fall vs non-fall).
    """
    
    def __init__(self, input_dim: int, hidden_dim: Optional[int] = None, dropout: float = 0.1):
        """
        Initialize classifier.
        
        Args:
            input_dim: Dimension of input embeddings (CLIP embedding dimension).
            hidden_dim: Optional hidden layer dimension. If None, uses linear classifier.
            dropout: Dropout probability for hidden layer.
        """
        super(LinearClassifier, self).__init__()
        
        if hidden_dim is None:
            # Simple linear classifier
            self.classifier = nn.Linear(input_dim, 1)
            self.use_hidden = False
        else:
            # MLP with one hidden layer
            self.classifier = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1)
            )
            self.use_hidden = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input embeddings of shape [B, D].
        
        Returns:
            Logits of shape [B, 1].
        """
        return self.classifier(x)


def extract_video_embedding(
    video_path: str,
    vlm: VisionLanguageModel,
    num_frames: Optional[int] = None,
    config: Optional[Dict] = None
) -> torch.Tensor:
    """
    Extract a single embedding vector for a video.
    
    This function:
    1. Samples frames from the video
    2. Encodes frames using CLIP
    3. Averages frame embeddings to get a single video embedding
    
    Args:
        video_path: Path to video file.
        vlm: Initialized VisionLanguageModel instance.
        num_frames: Number of frames to sample. If None, uses config default.
        config: Configuration dictionary.
    
    Returns:
        Tensor of shape [D] with video embedding.
    """
    if num_frames is None:
        if config is not None:
            num_frames = get_num_frames(config)
        else:
            num_frames = 8
    
    # Load and preprocess video frames
    video_frames = get_clip_inputs_from_video(
        video_path=video_path,
        num_frames=num_frames,
        processor=vlm.processor
    )
    
    # Encode frames: [N, D]
    frame_embeddings = vlm.encode_images(video_frames, normalize=True)
    
    # Average over frames to get single video embedding: [D]
    video_embedding = frame_embeddings.mean(dim=0)
    
    return video_embedding


def build_feature_dataset(
    video_label_pairs: List[Tuple[str, int]],
    vlm: VisionLanguageModel,
    num_frames: Optional[int] = None,
    config: Optional[Dict] = None,
    show_progress: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build feature dataset from labeled videos.
    
    This function extracts CLIP embeddings for all videos and creates
    feature matrices X (embeddings) and y (labels).
    
    Args:
        video_label_pairs: List of (video_path, label) tuples.
                          Label should be 0 (non-fall) or 1 (fall).
        vlm: Initialized VisionLanguageModel instance.
        num_frames: Number of frames to sample per video.
        config: Configuration dictionary.
        show_progress: Whether to show progress bar.
    
    Returns:
        Tuple of (X, y) where:
        - X: Tensor of shape [N, D] with video embeddings
        - y: Tensor of shape [N] with labels
    """
    embeddings = []
    labels = []
    
    iterator = tqdm(video_label_pairs, desc="Extracting embeddings") if show_progress else video_label_pairs
    
    for video_path, label in iterator:
        try:
            embedding = extract_video_embedding(
                video_path=video_path,
                vlm=vlm,
                num_frames=num_frames,
                config=config
            )
            embeddings.append(embedding.cpu())
            labels.append(label)
        except Exception as e:
            logger.warning(f"Failed to process video {video_path}: {e}")
            continue
    
    if len(embeddings) == 0:
        raise ValueError("No valid videos processed. Check video paths and formats.")
    
    # Stack into tensors
    X = torch.stack(embeddings, dim=0)  # [N, D]
    y = torch.tensor(labels, dtype=torch.float32)  # [N]
    
    logger.info(f"Built dataset: {len(X)} samples, embedding dim: {X.shape[1]}")
    
    return X, y


def train_few_shot_classifier(
    train_videos: List[Tuple[str, int]],
    vlm: VisionLanguageModel,
    val_videos: Optional[List[Tuple[str, int]]] = None,
    config: Optional[Dict] = None,
    hidden_dim: Optional[int] = None,
    save_path: Optional[str] = None
) -> LinearClassifier:
    """
    Train a few-shot classifier on CLIP embeddings.
    
    This function:
    1. Extracts CLIP embeddings for all training videos
    2. Trains a linear classifier on these embeddings
    3. CLIP model weights remain frozen (not updated)
    
    Args:
        train_videos: List of (video_path, label) tuples for training.
                     Label: 0 (non-fall) or 1 (fall).
        vlm: Initialized VisionLanguageModel instance.
        val_videos: Optional validation set for early stopping.
        config: Configuration dictionary.
        hidden_dim: Optional hidden layer dimension. If None, uses linear classifier.
        save_path: Optional path to save trained classifier.
    
    Returns:
        Trained LinearClassifier instance.
    """
    # Get training config
    if config is None:
        train_config = {
            "learning_rate": 0.001,
            "epochs": 50,
            "batch_size": 8,
            "optimizer": "adam",
            "weight_decay": 0.0001,
            "early_stopping_patience": 10,
        }
    else:
        train_config = get_few_shot_config(config)
    
    logger.info("Building training dataset...")
    X_train, y_train = build_feature_dataset(
        train_videos,
        vlm=vlm,
        config=config,
        show_progress=True
    )
    
    # Build validation set if provided
    if val_videos is not None:
        logger.info("Building validation dataset...")
        X_val, y_val = build_feature_dataset(
            val_videos,
            vlm=vlm,
            config=config,
            show_progress=True
        )
    else:
        X_val, y_val = None, None
    
    # Initialize classifier
    embedding_dim = X_train.shape[1]
    classifier = LinearClassifier(
        input_dim=embedding_dim,
        hidden_dim=hidden_dim,
        dropout=0.1
    )
    
    # Setup optimizer
    optimizer_name = train_config["optimizer"].lower()
    if optimizer_name == "adam":
        optimizer = optim.Adam(
            classifier.parameters(),
            lr=train_config["learning_rate"],
            weight_decay=train_config["weight_decay"]
        )
    elif optimizer_name == "sgd":
        optimizer = optim.SGD(
            classifier.parameters(),
            lr=train_config["learning_rate"],
            weight_decay=train_config["weight_decay"],
            momentum=0.9
        )
    elif optimizer_name == "adamw":
        optimizer = optim.AdamW(
            classifier.parameters(),
            lr=train_config["learning_rate"],
            weight_decay=train_config["weight_decay"]
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    # Loss function (binary cross-entropy)
    criterion = nn.BCEWithLogitsLoss()
    
    # Create data loaders
    train_dataset = VideoEmbeddingDataset(X_train, y_train)
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config["batch_size"],
        shuffle=True
    )
    
    if X_val is not None:
        val_dataset = VideoEmbeddingDataset(X_val, y_val)
        val_loader = DataLoader(
            val_dataset,
            batch_size=train_config["batch_size"],
            shuffle=False
        )
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    epochs = train_config["epochs"]
    
    logger.info(f"Training classifier for {epochs} epochs...")
    
    for epoch in range(epochs):
        # Training phase
        classifier.train()
        train_loss = 0.0
        
        for embeddings_batch, labels_batch in train_loader:
            optimizer.zero_grad()
            
            # Forward pass
            logits = classifier(embeddings_batch).squeeze(-1)
            loss = criterion(logits, labels_batch)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation phase
        if X_val is not None:
            classifier.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for embeddings_batch, labels_batch in val_loader:
                    logits = classifier(embeddings_batch).squeeze(-1)
                    loss = criterion(logits, labels_batch)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            
            logger.info(
                f"Epoch {epoch+1}/{epochs} - "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
            )
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                if save_path:
                    save_classifier(classifier, save_path)
            else:
                patience_counter += 1
                if patience_counter >= train_config["early_stopping_patience"]:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        else:
            logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}")
    
    # Load best model if validation was used
    if X_val is not None and save_path and os.path.exists(save_path):
        classifier = load_classifier(save_path, embedding_dim, hidden_dim)
        logger.info("Loaded best model from checkpoint")
    
    logger.info("Training completed!")
    
    return classifier


def few_shot_predict(
    video_path: str,
    vlm: VisionLanguageModel,
    classifier: LinearClassifier,
    threshold: float = 0.5,
    num_frames: Optional[int] = None,
    config: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Predict fall probability for a video using trained classifier.
    
    Args:
        video_path: Path to video file.
        vlm: Initialized VisionLanguageModel instance.
        classifier: Trained LinearClassifier instance.
        threshold: Threshold for binary classification (default: 0.5).
        num_frames: Number of frames to sample.
        config: Configuration dictionary.
    
    Returns:
        Dictionary with:
        - "fall_probability": Probability of fall (0-1).
        - "is_fall": Boolean indicating if fall was detected.
    """
    # Extract video embedding
    video_embedding = extract_video_embedding(
        video_path=video_path,
        vlm=vlm,
        num_frames=num_frames,
        config=config
    )
    
    # Get prediction from classifier
    classifier.eval()
    with torch.no_grad():
        logit = classifier(video_embedding.unsqueeze(0)).squeeze(-1)
        probability = torch.sigmoid(logit).item()
    
    is_fall = probability >= threshold
    
    return {
        "fall_probability": probability,
        "is_fall": is_fall,
    }


def save_classifier(classifier: LinearClassifier, save_path: str) -> None:
    """
    Save trained classifier to disk.
    
    Args:
        classifier: Trained LinearClassifier instance.
        save_path: Path to save the classifier.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(classifier.state_dict(), save_path)
    logger.info(f"Classifier saved to {save_path}")


def load_classifier(
    load_path: str,
    input_dim: int,
    hidden_dim: Optional[int] = None
) -> LinearClassifier:
    """
    Load trained classifier from disk.
    
    Args:
        load_path: Path to saved classifier.
        input_dim: Input embedding dimension (must match training).
        hidden_dim: Hidden layer dimension (must match training).
    
    Returns:
        Loaded LinearClassifier instance.
    """
    classifier = LinearClassifier(input_dim=input_dim, hidden_dim=hidden_dim)
    classifier.load_state_dict(torch.load(load_path, map_location="cpu"))
    logger.info(f"Classifier loaded from {load_path}")
    return classifier











