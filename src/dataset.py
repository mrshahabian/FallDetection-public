"""
PyTorch Dataset classes for skeleton action recognition
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Optional, Tuple, Dict, Any
import random


class SkeletonKTHDataset(Dataset):
    """
    Dataset class for KTH skeleton action recognition.
    
    Supports different model types:
    - 3D CNN: Returns [C=2, J=17, T=32]
    - 2D CNN: Returns [C=1, H=T, W=2J] image representation
    - ViT: Returns [C=1, H=T, W=2J] image representation (same as 2D CNN)
    """
    
    def __init__(self, data_root: str, split: str = 'train',
                 clip_len: int = 32, model_type: str = '3dcnn_simple',
                 transform: Optional[Any] = None, file_list: Optional[List[str]] = None):
        """
        Initialize dataset.
        
        Args:
            data_root: Root directory containing preprocessed .npz files
            split: 'train', 'val', or 'test'
            clip_len: Expected clip length (should match preprocessing)
            model_type: '3dcnn_simple', '3dcnn_deep', '2dcnn_resnet', '2dcnn_lenet', '2dcnn', 'vit', 'stgcn', or 'tcnt'
            transform: Optional data augmentation transform
            file_list: Optional list of specific files to use (overrides split)
        """
        self.data_root = data_root
        self.split = split
        self.clip_len = clip_len
        self.model_type = model_type
        self.transform = transform
        
        # Load file list
        if file_list is not None:
            self.file_list = file_list
        else:
            # Get all .npz files in data_root
            self.file_list = [
                os.path.join(data_root, f) 
                for f in os.listdir(data_root) 
                if f.endswith('.npz')
            ]
        
        # Filter by split if metadata available
        # For now, assume all files are in the same directory
        # Split should be handled by preprocessing.py
        
        print(f"Loaded {len(self.file_list)} samples for {split} split")
    
    def __len__(self) -> int:
        return len(self.file_list)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Returns:
            Dictionary with 'skeleton' (tensor), 'label' (tensor), and metadata
        """
        # Load .npz file
        data = np.load(self.file_list[idx])
        
        skeleton = data['skeleton']  # Shape: (T, J, C)
        label = int(data['label'])
        
        # Apply transform if provided
        if self.transform:
            skeleton = self.transform(skeleton)
        
        # Convert to tensor and reshape based on model type
        if self.model_type in ['3dcnn_simple', '3dcnn_deep']:
            # 3D CNN: [C, J, T]
            skeleton_tensor = torch.from_numpy(skeleton).float()  # (T, J, C)
            skeleton_tensor = skeleton_tensor.permute(2, 1, 0)  # (C, J, T)
            
        elif self.model_type in ['2dcnn_resnet', '2dcnn_lenet', '2dcnn']:
            # 2D CNN: [C=1, H=T, W=2J]
            if 'image_2d' in data:
                image_2d = data['image_2d']  # (T, 2J)
            else:
                # Generate on the fly
                image_2d = self._create_2d_image(skeleton)
            skeleton_tensor = torch.from_numpy(image_2d).float()  # (T, 2J)
            skeleton_tensor = skeleton_tensor.unsqueeze(0)  # (1, T, 2J)
            
        elif self.model_type == 'vit':
            # ViT: [C=1, H=T, W=2J] (2D image like 2D CNN)
            if 'image_2d' in data:
                image_2d = data['image_2d']  # (T, 2J)
            else:
                # Generate on the fly
                image_2d = self._create_2d_image(skeleton)
            skeleton_tensor = torch.from_numpy(image_2d).float()  # (T, 2J)
            skeleton_tensor = skeleton_tensor.unsqueeze(0)  # (1, T, 2J)
        elif self.model_type in ['stgcn', 'tcnt']:
            # ST-GCN, TCNTE: [C, J, T]
            skeleton_tensor = torch.from_numpy(skeleton).float()  # (T, J, C)
            skeleton_tensor = skeleton_tensor.permute(2, 1, 0)  # (C, J, T)
            
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")
        
        return {
            'skeleton': skeleton_tensor,
            'label': torch.tensor(label, dtype=torch.long),
            'file_path': self.file_list[idx]
        }
    
    def _create_2d_image(self, skeleton: np.ndarray) -> np.ndarray:
        """Create 2D image representation from skeleton."""
        T, J, C = skeleton.shape
        x_coords = skeleton[:, :, 0]  # (T, J)
        y_coords = skeleton[:, :, 1]  # (T, J)
        image_2d = np.concatenate([x_coords, y_coords], axis=1)  # (T, 2J)
        return image_2d


def get_dataloaders(config: Dict[str, Any], 
                   train_files: Optional[List[str]] = None,
                   val_files: Optional[List[str]] = None,
                   test_files: Optional[List[str]] = None) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test DataLoaders.
    
    Args:
        config: Configuration dictionary with dataset and training settings
        train_files: Optional list of training file paths
        val_files: Optional list of validation file paths
        test_files: Optional list of test file paths
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Get paths from config
    data_config = config.get('dataset', {})
    training_config = config.get('training', {})
    model_config = config.get('model', {})
    
    data_root = config.get('paths', {}).get('skeleton_data_root', './data')
    model_type = model_config.get('type', '3dcnn_simple')
    
    # Determine which skeleton source to use
    skeleton_source = model_config.get('skeleton_source', 'openpose')
    
    # Priority order for preprocessed path:
    # 1. training.preprocessed_path (highest priority)
    # 2. extraction.preprocessed_path
    # 3. Auto-detect: {skeleton_data_root}/{skeleton_source}/preprocessed
    # 4. Fallback: {skeleton_data_root}/{skeleton_source} (raw data)
    
    preprocessed_path = None
    
    # Check training config first (highest priority)
    if training_config.get('preprocessed_path'):
        preprocessed_path = training_config.get('preprocessed_path')
        if os.path.exists(preprocessed_path):
            data_root = preprocessed_path
            print(f"Using preprocessed directory from training config: {data_root}")
        else:
            print(f"Warning: preprocessed_path in training config not found: {preprocessed_path}")
            preprocessed_path = None  # Fall through to next option
    
    # Check extraction config if training config didn't specify
    if preprocessed_path is None:
        extraction_config = config.get('extraction', {})
        if extraction_config.get('preprocessed_path'):
            preprocessed_path = extraction_config.get('preprocessed_path')
            if os.path.exists(preprocessed_path):
                data_root = preprocessed_path
                print(f"Using preprocessed directory from extraction config: {data_root}")
            else:
                print(f"Warning: preprocessed_path in extraction config not found: {preprocessed_path}")
                preprocessed_path = None  # Fall through to auto-detect
    
    # Auto-detect if no custom path specified
    if preprocessed_path is None:
        auto_preprocessed_path = os.path.join(data_root, skeleton_source, 'preprocessed')
        if os.path.exists(auto_preprocessed_path):
            data_root = auto_preprocessed_path
            print(f"Using auto-detected preprocessed directory: {data_root}")
        else:
            # Fallback: try skeleton source root directly (raw data)
            alt_path = os.path.join(data_root, skeleton_source)
            if os.path.exists(alt_path):
                data_root = alt_path
                print(f"Using raw skeleton directory (no preprocessed data found): {data_root}")
            else:
                # Use data_root as-is
                print(f"Using data root as-is: {data_root}")
                pass
    
    # If file lists not provided, use split_dataset
    if train_files is None or val_files is None or test_files is None:
        import sys
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from preprocessing import split_dataset
        
        # Check if data_root exists and has files
        if not os.path.exists(data_root):
            raise ValueError(
                f"Data directory does not exist: {data_root}\n"
                f"  Please check your config paths in base_config.yaml\n"
                f"  Expected skeleton source: {skeleton_source}"
            )
        
        # Count .npz files
        npz_files = [f for f in os.listdir(data_root) if f.endswith('.npz')] if os.path.isdir(data_root) else []
        
        if len(npz_files) == 0:
            # Check if raw data exists (needs preprocessing)
            raw_data_path = os.path.join(config.get('paths', {}).get('skeleton_data_root', './data'), skeleton_source)
            if os.path.exists(raw_data_path):
                raise ValueError(
                    f"No preprocessed .npz files found in: {data_root}\n"
                    f"  Found raw data at: {raw_data_path}\n"
                    f"  You need to run preprocessing first:\n"
                    f"    python -m src.preprocessing preprocess_kth_skeletons {raw_data_path} {data_root}"
                )
            else:
                raise ValueError(
                    f"No data found!\n"
                    f"  Checked: {data_root}\n"
                    f"  Skeleton source: {skeleton_source}\n"
                    f"  Raw data path: {raw_data_path}\n"
                    f"\nPossible solutions:\n"
                    f"  1. Use correct skeleton source: --skeleton_source yolov11\n"
                    f"  2. Run skeleton extraction if data doesn't exist\n"
                    f"  3. Run preprocessing to generate .npz files"
                )
        
        split_strategy = data_config.get('split_strategy', 'random')
        train_files, val_files, test_files = split_dataset(
            data_root,
            split_strategy=split_strategy,
            train_ratio=data_config.get('train_ratio', 0.7),
            val_ratio=data_config.get('val_ratio', 0.15),
            test_ratio=data_config.get('test_ratio', 0.15),
            seed=config.get('seed', 42)
        )
        
        # Option to combine train+val during training (recommended for random split)
        # Test set is always kept separate for final evaluation
        combine_train_val = data_config.get('combine_train_val', False)
        if combine_train_val and len(train_files) > 0 and len(val_files) > 0:
            print(f"\nCombining train+val sets for training (test set kept separate for evaluation)")
            print(f"  Before: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")
            
            # Combine train and val into a larger pool
            combined_files = train_files + val_files
            
            # Split the combined set into new train/val for monitoring
            # Use 85% for training, 15% for validation monitoring
            random.seed(config.get('seed', 42))
            random.shuffle(combined_files)
            
            split_point = int(len(combined_files) * 0.85)
            # Ensure we have at least some validation samples
            if split_point == len(combined_files):
                split_point = max(1, len(combined_files) - 1)
            
            train_files = combined_files[:split_point]
            val_files = combined_files[split_point:]
            
            print(f"  After:  Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")
            print(f"  Note: Combined train+val split into new train/val. Test set ({len(test_files)} samples) kept separate for final evaluation")
    
    # Get clip length from extraction config or default
    extraction_config = config.get('extraction', {})
    clip_len = extraction_config.get('clip_length', 32)
    
    # Create datasets
    train_dataset = SkeletonKTHDataset(
        data_root=data_root,
        split='train',
        clip_len=clip_len,
        model_type=model_type,
        file_list=train_files
    )
    
    val_dataset = SkeletonKTHDataset(
        data_root=data_root,
        split='val',
        clip_len=clip_len,
        model_type=model_type,
        file_list=val_files
    )
    
    test_dataset = SkeletonKTHDataset(
        data_root=data_root,
        split='test',
        clip_len=clip_len,
        model_type=model_type,
        file_list=test_files
    )
    
    # Validate datasets are not empty
    if len(train_dataset) == 0:
        raise ValueError(
            f"No training data found!\n"
            f"  Data root: {data_root}\n"
            f"  Skeleton source: {skeleton_source}\n"
            f"  Model type: {model_type}\n"
            f"\nPossible solutions:\n"
            f"  1. Check if data exists at: {data_root}\n"
            f"  2. Verify skeleton source matches your data (try: --skeleton_source yolov11)\n"
            f"  3. Run preprocessing to generate .npz files\n"
            f"  4. Check config paths in base_config.yaml"
        )
    
    if len(val_dataset) == 0 and not data_config.get('combine_train_val', False):
        print(f"⚠ Warning: Validation set is empty ({len(val_dataset)} samples)")
        print(f"  Consider setting combine_train_val: true in config")
    
    if len(test_dataset) == 0:
        print(f"⚠ Warning: Test set is empty ({len(test_dataset)} samples)")
        print(f"  Final evaluation will not be possible")
    
    # Get DataLoader parameters
    batch_size = training_config.get('batch_size', 16)
    num_workers = training_config.get('num_workers', 4)
    pin_memory = training_config.get('pin_memory', True)
    shuffle_train = training_config.get('shuffle_train', True)
    
    # Create DataLoaders (only if dataset is not empty)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train if len(train_dataset) > 0 else False,
        num_workers=num_workers,
        pin_memory=pin_memory
    ) if len(train_dataset) > 0 else None
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    ) if len(val_dataset) > 0 else None
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    ) if len(test_dataset) > 0 else None
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test dataset
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from utils import load_config, merge_configs
    
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
        config = load_config(config_path)
        
        train_loader, val_loader, test_loader = get_dataloaders(config)
        
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
        print(f"Test batches: {len(test_loader)}")
        
        # Test a batch
        for batch in train_loader:
            print(f"Batch skeleton shape: {batch['skeleton'].shape}")
            print(f"Batch label shape: {batch['label'].shape}")
            break

