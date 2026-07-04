"""
Train CNN models (2D CNN, 3D CNN, ViT) on fall videos for binary classification.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models import create_model
from src.skeleton_extraction import extract_with_yolov11
from src.preprocessing import generate_clips
from evaluation.video_sampler import get_video_files

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FallVideoDataset(Dataset):
    """Dataset for fall detection from videos."""
    
    def __init__(self, video_paths: list, labels: list, model_type: str, 
                 window_length: int = 32, overlap: float = 0.5):
        """
        Initialize dataset.
        
        Args:
            video_paths: List of video file paths
            labels: List of labels (0=non-fall, 1=fall)
            model_type: Type of model ('2dcnn_resnet', '2dcnn_lenet', '3dcnn_simple', '3dcnn_deep', 'vit')
            window_length: Length of each window in frames
            overlap: Overlap ratio between windows
        """
        self.video_paths = video_paths
        self.labels = labels
        self.model_type = model_type
        self.window_length = window_length
        self.overlap = overlap
        
        # Extract skeletons and generate clips
        self.clips = []
        self.clip_labels = []
        
        logger.info(f"Processing {len(video_paths)} videos...")
        for video_path, label in tqdm(zip(video_paths, labels), total=len(video_paths), desc="Extracting skeletons"):
            try:
                result = extract_with_yolov11(video_path)
                if isinstance(result, tuple):
                    skeletons, _ = result  # Extract keypoints, ignore bboxes
                else:
                    skeletons = result
                if skeletons is None or len(skeletons) == 0:
                    continue
                
                # Generate clips
                clips = generate_clips(skeletons, clip_length=window_length, overlap=overlap)
                
                for clip in clips:
                    self.clips.append(clip)
                    self.clip_labels.append(label)
            except Exception as e:
                logger.warning(f"Error processing {video_path}: {e}")
                continue
        
        logger.info(f"Generated {len(self.clips)} clips from {len(video_paths)} videos")
    
    def __len__(self):
        return len(self.clips)
    
    def __getitem__(self, idx):
        clip = self.clips[idx]
        label = self.clip_labels[idx]
        
        # Convert to tensor and prepare for model
        if self.model_type in ['3dcnn_simple', '3dcnn_deep']:
            # Shape: [C=2, J=17, T=32]
            # clip is [T, J, 2] -> transpose to [2, J, T]
            if clip.ndim == 3:
                clip = np.transpose(clip, (2, 1, 0))  # [T, J, 2] -> [2, J, T]
            tensor = torch.from_numpy(clip).float()
        elif self.model_type in ['2dcnn_resnet', '2dcnn_lenet', 'vit']:
            # Reshape to 2D image: [T, J*2]
            if clip.ndim == 3:
                T, J, C = clip.shape
                clip = clip.reshape(T, J * C)
            tensor = torch.from_numpy(clip).float()
            tensor = tensor.unsqueeze(0)  # Add channel dimension: [1, T, J*2]
        else:
            tensor = torch.from_numpy(clip).float()
        
        return {'skeleton': tensor, 'label': label}


def prepare_dataset(fall_base_dir: str, non_fall_videos_dir: str = None,
                   train_ratio: float = 0.8, seed: int = 42, 
                   num_fall: int = 100, num_non_fall: int = 100):
    """
    Prepare train/val split from videos using balanced sampling.
    
    Args:
        fall_base_dir: Base directory containing fall videos (may have Raw_Video and Raw_Video_part2)
        non_fall_videos_dir: Directory containing non-fall videos
        train_ratio: Ratio for train/val split
        seed: Random seed
        num_fall: Number of fall videos to sample
        num_non_fall: Number of non-fall videos to sample
    
    Returns:
        Tuple of (train_videos, train_labels, val_videos, val_labels)
    """
    from evaluation.dataset_sampler import get_all_fall_videos, get_video_files
    import random
    random.seed(seed)
    
    # Get all fall videos from multiple directories
    # Use single-person dataset only (best fitted process)
    logger.info("Collecting fall videos (single-person dataset only)...")
    all_fall_videos = get_all_fall_videos(fall_base_dir, single_person_only=True)
    
    # Get non-fall videos
    if non_fall_videos_dir and os.path.exists(non_fall_videos_dir):
        logger.info("Collecting non-fall videos...")
        all_non_fall_videos = get_video_files(non_fall_videos_dir)
    else:
        logger.warning("No non-fall videos directory. Using fall videos only.")
        all_non_fall_videos = []
    
    # Sample videos
    if len(all_fall_videos) < num_fall:
        sampled_fall = all_fall_videos
    else:
        sampled_fall = random.sample(all_fall_videos, num_fall)
    
    if len(all_non_fall_videos) < num_non_fall:
        sampled_non_fall = all_non_fall_videos
    else:
        sampled_non_fall = random.sample(all_non_fall_videos, num_non_fall)
    
    # Split into train/val
    random.shuffle(sampled_fall)
    random.shuffle(sampled_non_fall)
    
    train_fall = sampled_fall[:int(len(sampled_fall) * train_ratio)]
    val_fall = sampled_fall[int(len(sampled_fall) * train_ratio):]
    
    train_non_fall = sampled_non_fall[:int(len(sampled_non_fall) * train_ratio)] if sampled_non_fall else []
    val_non_fall = sampled_non_fall[int(len(sampled_non_fall) * train_ratio):] if sampled_non_fall else []
    
    train_videos = train_fall + train_non_fall
    train_labels = [1] * len(train_fall) + [0] * len(train_non_fall)
    
    val_videos = val_fall + val_non_fall
    val_labels = [1] * len(val_fall) + [0] * len(val_non_fall)
    
    # Shuffle
    combined = list(zip(train_videos, train_labels))
    random.shuffle(combined)
    train_videos, train_labels = zip(*combined)
    
    combined = list(zip(val_videos, val_labels))
    random.shuffle(combined)
    val_videos, val_labels = zip(*combined)
    
    logger.info(f"Train: {len(train_videos)} videos ({sum(train_labels)} fall, {len(train_labels) - sum(train_labels)} non-fall)")
    logger.info(f"Val: {len(val_videos)} videos ({sum(val_labels)} fall, {len(val_labels) - sum(val_labels)} non-fall)")
    
    return list(train_videos), list(train_labels), list(val_videos), list(val_labels)


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch in tqdm(train_loader, desc="Training", leave=False):
        skeletons = batch['skeleton'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        outputs = model(skeletons)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """Validate model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating", leave=False):
            skeletons = batch['skeleton'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(skeletons)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(val_loader) if len(val_loader) > 0 else 0.0
    epoch_acc = 100 * correct / total if total > 0 else 0.0
    return epoch_loss, epoch_acc


def main():
    parser = argparse.ArgumentParser(description="Train CNN models for fall detection")
    parser.add_argument("--model_type", type=str, required=True,
                       choices=['2dcnn_resnet', '2dcnn_lenet', '3dcnn_simple', '3dcnn_deep', 'vit'],
                       help="Type of model to train")
    parser.add_argument("--fall_base_dir", type=str,
                       default="/home/reza/Documents/Datasets/Fall/Fall",
                       help="Base directory containing fall videos (may have Raw_Video and Raw_Video_part2)")
    parser.add_argument("--kth_dir", type=str,
                       default="/home/reza/Documents/Datasets/KTH",
                       help="KTH dataset directory (used as non-fall samples)")
    parser.add_argument("--non_fall_videos_dir", type=str, default=None,
                       help="Directory containing non-fall videos (optional, KTH is preferred)")
    parser.add_argument("--num_fall", type=int, default=100,
                       help="Number of fall videos to sample for training")
    parser.add_argument("--num_non_fall", type=int, default=100,
                       help="Number of non-fall videos to sample for training")
    parser.add_argument("--epochs", type=int, default=50,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                       help="Learning rate")
    parser.add_argument("--window_length", type=int, default=32,
                       help="Window length in frames")
    parser.add_argument("--overlap", type=float, default=0.5,
                       help="Overlap ratio between windows")
    parser.add_argument("--save_path", type=str,
                       default=None,
                       help="Path to save checkpoint")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use")
    
    args = parser.parse_args()
    
    # Set device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    
    # Prepare dataset using KTH for non-fall
    from evaluation.dataset_sampler import get_kth_videos
    from evaluation.video_sampler import get_video_files
    import random
    
    random.seed(42)
    
    # Get fall videos (single-person only)
    from evaluation.dataset_sampler import get_all_fall_videos
    all_fall_videos = get_all_fall_videos(args.fall_base_dir, single_person_only=True)
    logger.info(f"Using single-person dataset only (Raw_Video)")
    
    # Get non-fall videos from KTH
    if args.kth_dir and os.path.exists(args.kth_dir):
        all_non_fall_videos = get_kth_videos(args.kth_dir)
        logger.info(f"Using KTH dataset as non-fall samples: {len(all_non_fall_videos)} videos")
    elif args.non_fall_videos_dir and os.path.exists(args.non_fall_videos_dir):
        all_non_fall_videos = get_video_files(args.non_fall_videos_dir)
    else:
        logger.warning("No KTH or non-fall directory found!")
        all_non_fall_videos = []
    
    # Sample videos
    sampled_fall = random.sample(all_fall_videos, min(args.num_fall, len(all_fall_videos))) if len(all_fall_videos) >= args.num_fall else all_fall_videos
    sampled_non_fall = random.sample(all_non_fall_videos, min(args.num_non_fall, len(all_non_fall_videos))) if len(all_non_fall_videos) >= args.num_non_fall else all_non_fall_videos
    
    # Split into train/val
    random.shuffle(sampled_fall)
    random.shuffle(sampled_non_fall)
    
    train_fall = sampled_fall[:int(len(sampled_fall) * 0.8)]
    val_fall = sampled_fall[int(len(sampled_fall) * 0.8):]
    
    train_non_fall = sampled_non_fall[:int(len(sampled_non_fall) * 0.8)] if sampled_non_fall else []
    val_non_fall = sampled_non_fall[int(len(sampled_non_fall) * 0.8):] if sampled_non_fall else []
    
    train_videos = train_fall + train_non_fall
    train_labels = [1] * len(train_fall) + [0] * len(train_non_fall)
    
    val_videos = val_fall + val_non_fall
    val_labels = [1] * len(val_fall) + [0] * len(val_non_fall)
    
    # Create datasets
    train_dataset = FallVideoDataset(train_videos, train_labels, args.model_type,
                                     args.window_length, args.overlap)
    val_dataset = FallVideoDataset(val_videos, val_labels, args.model_type,
                                   args.window_length, args.overlap)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Create model (binary classification)
    model = create_model(args.model_type, num_classes=2)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Training loop
    best_val_acc = 0.0
    save_path = args.save_path or f"evaluation/checkpoints/{args.model_type}_fall_detection.pt"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    for epoch in range(args.epochs):
        logger.info(f"\nEpoch {epoch+1}/{args.epochs}")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_type': args.model_type,
                'epoch': epoch,
                'val_acc': val_acc,
                'num_classes': 2
            }, save_path)
            logger.info(f"Saved best model to {save_path}")
    
    logger.info(f"Training completed! Best validation accuracy: {best_val_acc:.2f}%")


if __name__ == "__main__":
    main()

