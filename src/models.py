"""
Model architectures for skeleton-based action recognition
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class Skeleton3DCNN_Simple(nn.Module):
    """
    Simple 3D CNN for skeleton sequences.
    
    Input shape: [B, C=2, J=17, T=32]
    Output shape: [B, num_classes=6]
    """
    
    def __init__(self, num_classes: int = 6, dropout: float = 0.5):
        """
        Initialize model.
        
        Args:
            num_classes: Number of action classes
            dropout: Dropout probability
        """
        super(Skeleton3DCNN_Simple, self).__init__()
        
        # First conv block
        self.conv1 = nn.Conv3d(
            in_channels=2,
            out_channels=32,
            kernel_size=(3, 3, 3),
            stride=(1, 1, 1),
            padding=(1, 1, 1)
        )
        self.bn1 = nn.BatchNorm3d(32)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))  # Don't pool depth
        
        # Second conv block
        self.conv2 = nn.Conv3d(
            in_channels=32,
            out_channels=64,
            kernel_size=(3, 3, 3),
            stride=(1, 1, 1),
            padding=(1, 1, 1)
        )
        self.bn2 = nn.BatchNorm3d(64)
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))  # Don't pool depth
        
        # Fully connected layers
        # After pooling: [B, 64, 1, 4, 8] -> flatten -> 64 * 1 * 4 * 8 = 2048
        self.fc1 = nn.Linear(64 * 1 * 4 * 8, 512)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape [B, 2, 17, 32]
            
        Returns:
            Logits of shape [B, num_classes]
        """
        # Add a dimension to make it 5D for Conv3d: [B, C, D, H, W]
        # [B, 2, 17, 32] -> [B, 2, 1, 17, 32]
        x = x.unsqueeze(2)  # [B, 2, 1, 17, 32]
        
        # Conv block 1
        x = self.conv1(x)  # [B, 32, 1, 17, 32]
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)  # [B, 32, 1, 8, 16]
        
        # Conv block 2
        x = self.conv2(x)  # [B, 64, 1, 8, 16]
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)  # [B, 64, 1, 4, 8]
        
        # Flatten
        x = x.view(x.size(0), -1)  # [B, 64 * 1 * 4 * 8]
        
        # Fully connected
        x = self.fc1(x)  # [B, 512]
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)  # [B, num_classes]
        
        return x


class ResidualBlock3D(nn.Module):
    """3D Residual block for deep 3D CNN."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super(ResidualBlock3D, self).__init__()
        
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 3), 
                               stride=(1, 1, 1), padding=(1, 1, 1))
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=(3, 3, 3),
                               stride=(1, 1, 1), padding=(1, 1, 1))
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        # Shortcut connection
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1),
                nn.BatchNorm3d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out


class Skeleton3DCNN_Deep(nn.Module):
    """
    Dense 3D CNN with extensive residual connections for skeleton sequences.
    Memory-optimized dense architecture with 15 residual blocks to capture rich spatial-temporal features.
    
    Input shape: [B, C=2, J=17, T=32]
    Output shape: [B, num_classes=6]
    
    Architecture: 15 residual blocks (64→128→256→512 channels) + 4 dense FC layers (2048→1024→512→256)
    Memory optimized: Reduced from 17 blocks and 5 FC layers to fit in GPU memory while maintaining density.
    """
    
    def __init__(self, num_classes: int = 6, 
                 dropout1: float = 0.5, dropout2: float = 0.4, dropout3: float = 0.3, 
                 dropout4: float = 0.2):
        """
        Initialize dense 3D CNN model (memory optimized).
        
        Args:
            num_classes: Number of action classes
            dropout1-4: Dropout probabilities for FC layers (decreasing)
        """
        super(Skeleton3DCNN_Deep, self).__init__()
        
        # Initial conv block with larger capacity
        self.conv1 = nn.Conv3d(2, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.bn1 = nn.BatchNorm3d(64)
        
        # Dense residual blocks - many more layers for rich feature extraction
        # Stage 1: 64 channels (3 blocks for dense spatial-temporal features)
        self.res_block1 = ResidualBlock3D(64, 64)
        self.res_block2 = ResidualBlock3D(64, 64)
        self.res_block3 = ResidualBlock3D(64, 64)
        
        # Stage 2: 128 channels (4 blocks)
        self.res_block4 = ResidualBlock3D(64, 128)
        self.res_block5 = ResidualBlock3D(128, 128)
        self.res_block6 = ResidualBlock3D(128, 128)
        self.res_block7 = ResidualBlock3D(128, 128)
        
        # Stage 3: 256 channels (4 blocks)
        self.res_block8 = ResidualBlock3D(128, 256)
        self.res_block9 = ResidualBlock3D(256, 256)
        self.res_block10 = ResidualBlock3D(256, 256)
        self.res_block11 = ResidualBlock3D(256, 256)
        
        # Stage 4: 512 channels (4 blocks) - reduced from 1024 to save memory
        self.res_block12 = ResidualBlock3D(256, 512)
        self.res_block13 = ResidualBlock3D(512, 512)
        self.res_block14 = ResidualBlock3D(512, 512)
        self.res_block15 = ResidualBlock3D(512, 512)
        
        # Pooling layers
        self.pool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))  # Don't pool depth
        self.adaptive_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # Dense fully connected layers (reduced sizes for memory efficiency while maintaining density)
        self.fc1 = nn.Linear(512, 2048)  # Reduced from 4096 to save memory
        self.dropout1 = nn.Dropout(dropout1)
        self.fc2 = nn.Linear(2048, 1024)  # Reduced from 2048
        self.dropout2 = nn.Dropout(dropout2)
        self.fc3 = nn.Linear(1024, 512)
        self.dropout3 = nn.Dropout(dropout3)
        self.fc4 = nn.Linear(512, 256)
        self.dropout4 = nn.Dropout(dropout4)
        self.fc5 = nn.Linear(256, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through dense 3D CNN.
        
        Args:
            x: Input tensor of shape [B, 2, 17, 32]
            
        Returns:
            Logits of shape [B, num_classes]
        """
        # Add a dimension to make it 5D for Conv3d: [B, C, D, H, W]
        # [B, 2, 17, 32] -> [B, 2, 1, 17, 32]
        x = x.unsqueeze(2)  # [B, 2, 1, 17, 32]
        
        # Initial conv with larger capacity
        x = F.relu(self.bn1(self.conv1(x)))  # [B, 64, 1, 17, 32]
        x = self.pool(x)  # [B, 64, 1, 8, 16]
        
        # Stage 1: Dense spatial-temporal feature extraction (64 channels, 3 blocks)
        x = self.res_block1(x)  # [B, 64, 1, 8, 16]
        x = self.res_block2(x)  # [B, 64, 1, 8, 16]
        x = self.res_block3(x)  # [B, 64, 1, 8, 16]
        x = self.pool(x)  # [B, 64, 1, 4, 8]
        
        # Stage 2: Deeper features (128 channels, 4 blocks)
        x = self.res_block4(x)  # [B, 128, 1, 4, 8]
        x = self.res_block5(x)  # [B, 128, 1, 4, 8]
        x = self.res_block6(x)  # [B, 128, 1, 4, 8]
        x = self.res_block7(x)  # [B, 128, 1, 4, 8]
        x = self.pool(x)  # [B, 128, 1, 2, 4]
        
        # Stage 3: Rich temporal-spatial patterns (256 channels, 4 blocks)
        x = self.res_block8(x)  # [B, 256, 1, 2, 4]
        x = self.res_block9(x)  # [B, 256, 1, 2, 4]
        x = self.res_block10(x)  # [B, 256, 1, 2, 4]
        x = self.res_block11(x)  # [B, 256, 1, 2, 4]
        x = self.pool(x)  # [B, 256, 1, 1, 2]
        
        # Stage 4: High-level features (512 channels, 4 blocks)
        x = self.res_block12(x)  # [B, 512, 1, 1, 2]
        x = self.res_block13(x)  # [B, 512, 1, 1, 2]
        x = self.res_block14(x)  # [B, 512, 1, 1, 2]
        x = self.res_block15(x)  # [B, 512, 1, 1, 2]
        # Skip pooling here - spatial dimensions are already 1x1, can't pool further
        
        # Adaptive pooling to fixed size (handles variable temporal dimension)
        x = self.adaptive_pool(x)  # [B, 512, 1, 1, 1]
        
        # Flatten
        x = x.view(x.size(0), -1)  # [B, 512]
        
        # Dense fully connected layers for rich feature combination
        x = F.relu(self.fc1(x))  # [B, 2048]
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))  # [B, 1024]
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))  # [B, 512]
        x = self.dropout3(x)
        x = F.relu(self.fc4(x))  # [B, 256]
        x = self.dropout4(x)
        x = self.fc5(x)  # [B, num_classes]
        
        return x


class CustomResNet(nn.Module):
    """
    Custom ResNet-18 based model for skeleton sequences.
    
    Input shape: [B, C=1, H=32, W=34] (H=frames, W=2*J where J=17 joints)
    Output shape: [B, num_classes]
    """
    
    def __init__(self, num_classes: int = 6, pretrained: bool = True):
        """
        Initialize model.
        
        Args:
            num_classes: Number of action classes
            pretrained: Whether to use pretrained ResNet-18 weights
        """
        super(CustomResNet, self).__init__()
        
        # Import ResNet
        from torchvision import models
        
        # Load the ResNet-18 model
        self.resnet = models.resnet18(pretrained=pretrained)
        
        # Modify the first layer to accept 1 input channel
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Modify the output layer to have the desired number of classes
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape [B, 1, 32, 34]
            
        Returns:
            Logits of shape [B, num_classes]
        """
        return self.resnet(x)


class LeNetRegularized(nn.Module):
    """
    LeNet architecture with regularization for skeleton sequences.
    
    Input shape: [B, C=1, H=32, W=34] (H=frames, W=2*J where J=17 joints)
    Output shape: [B, num_classes]
    """
    
    def __init__(self, numChannels: int = 1, classes: int = 6, dropout_rate: float = 0.5, weight_decay: float = 0.001):
        """
        Initialize model.
        
        Args:
            numChannels: Number of input channels (1 for skeleton data)
            classes: Number of action classes
            dropout_rate: Dropout probability
            weight_decay: L2 regularization weight
        """
        super(LeNetRegularized, self).__init__()
        
        self.dropout_rate = dropout_rate
        self.classes = classes
        self.weight_decay = weight_decay
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(numChannels, 10, kernel_size=3)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=3)
        
        # Calculate FC input size based on input shape (32, 34)
        # After conv1 (3x3): (32-3+1, 34-3+1) = (30, 32)
        # After maxpool (2x2): (15, 16)
        # After conv2 (3x3): (15-3+1, 16-3+1) = (13, 14)
        # After maxpool (2x2): (6, 7)
        # FC input: 20 * 6 * 7 = 840
        self.fc1 = nn.Linear(840, 500)
        self.fc2 = nn.Linear(500, 250)
        self.fc3 = nn.Linear(250, classes)
        
        self.dropout = nn.Dropout(dropout_rate)
        # Note: LogSoftmax removed for compatibility with CrossEntropyLoss
        # Original architecture used LogSoftmax with NLLLoss
        # If you want to use LogSoftmax, switch to NLLLoss in training
        # self.logSoftmax = nn.LogSoftmax(dim=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape [B, 1, 32, 34]
            
        Returns:
            Logits of shape [B, num_classes]
        """
        x = F.max_pool2d(F.relu(self.conv1(x)), 2, 2)
        x = self.dropout(x)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2, 2)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # LogSoftmax removed for compatibility with CrossEntropyLoss
        # If using NLLLoss, uncomment: x = self.logSoftmax(x)
        return x
    
    def l2_regularization_loss(self):
        """
        Calculate L2 regularization loss.
        
        Returns:
            L2 regularization loss
        """
        l2_loss = 0
        for param in self.parameters():
            l2_loss += torch.norm(param)
        return l2_loss * self.weight_decay


class PatchEmbedding(nn.Module):
    """Patch embedding for Vision Transformer (2D image input)."""
    
    def __init__(self, img_size: Tuple[int, int] = (32, 34), patch_size: Tuple[int, int] = (4, 4),
                 in_channels: int = 1, embed_dim: int = 128):
        """
        Initialize patch embedding for 2D image input.
        
        Args:
            img_size: (H, W) image size
            patch_size: (patch_h, patch_w) patch size
            in_channels: Number of input channels (1 for skeleton 2D image)
            embed_dim: Embedding dimension
        """
        super(PatchEmbedding, self).__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        
        H, W = img_size
        patch_h, patch_w = patch_size
        
        # Calculate number of patches
        self.num_patches_h = H // patch_h
        self.num_patches_w = W // patch_w
        self.num_patches = self.num_patches_h * self.num_patches_w
        
        # Patch projection: Conv2d to embed patches
        self.proj = nn.Conv2d(in_channels, embed_dim, 
                             kernel_size=patch_size,
                             stride=patch_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape [B, 1, H, W] (2D image)
            
        Returns:
            Patches of shape [B, num_patches, embed_dim]
        """
        # x: [B, 1, H, W]
        x = self.proj(x)  # [B, embed_dim, num_patches_h, num_patches_w]
        B, C, H, W = x.shape
        
        # Flatten spatial dimensions: [B, embed_dim, num_patches_h, num_patches_w] -> [B, embed_dim, num_patches]
        x = x.flatten(2)  # [B, embed_dim, num_patches]
        x = x.transpose(1, 2)  # [B, num_patches, embed_dim]
        
        return x


class TransformerBlock(nn.Module):
    """Enhanced Transformer encoder block with robust multi-head attention."""
    
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 dropout: float = 0.1, attention_dropout: float = 0.1,
                 use_pre_norm: bool = True):
        """
        Initialize enhanced transformer block.
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads (increased for robustness)
            mlp_ratio: MLP hidden dimension ratio
            dropout: Dropout probability
            attention_dropout: Attention dropout probability
            use_pre_norm: Use pre-normalization (True) or post-normalization (False)
        """
        super(TransformerBlock, self).__init__()
        
        self.use_pre_norm = use_pre_norm
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Enhanced multi-head attention with more heads for robustness
        self.attn = nn.MultiheadAttention(
            embed_dim, 
            num_heads, 
            
            dropout=attention_dropout,
            batch_first=True,
            bias=True  # Enable bias for better learning
        )
        
        # MLP with GELU activation
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
        # Dropout for residual connections
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with enhanced attention mechanism.
        
        Args:
            x: Input tensor of shape [B, N, D]
            
        Returns:
            Output tensor of shape [B, N, D]
        """
        if self.use_pre_norm:
            # Pre-normalization (more stable training)
            # Self-attention with residual connection
            x_norm = self.norm1(x)
            attn_out, attn_weights = self.attn(x_norm, x_norm, x_norm)
            attn_out = self.dropout1(attn_out)
            x = x + attn_out  # Residual connection
            
            # MLP with residual connection
            x_norm = self.norm2(x)
            mlp_out = self.mlp(x_norm)
            mlp_out = self.dropout2(mlp_out)
            x = x + mlp_out  # Residual connection
        else:
            # Post-normalization (original ViT style)
            # Self-attention
            attn_out, attn_weights = self.attn(x, x, x)
            attn_out = self.dropout1(attn_out)
            x = x + attn_out
            x = self.norm1(x)
            
            # MLP
            mlp_out = self.mlp(x)
            mlp_out = self.dropout2(mlp_out)
            x = x + mlp_out
            x = self.norm2(x)
        
        return x


class SkeletonViT(nn.Module):
    """
    Enhanced Vision Transformer for skeleton sequences with 2D image input.
    
    Input shape: [B, 1, H=32, W=34] (2D image representation like 2D CNN)
    Output shape: [B, num_classes=6]
    """
    
    def __init__(self, img_size: Tuple[int, int] = (32, 34),
                 patch_size: Tuple[int, int] = (4, 4),
                 in_channels: int = 1,
                 embed_dim: int = 128,
                 num_layers: int = 6,  # Increased default layers for robustness
                 num_heads: int = 8,  # Increased default heads for robust attention
                 mlp_ratio: float = 4.0,
                 dropout: float = 0.1,
                 attention_dropout: float = 0.1,
                 num_classes: int = 6,
                 use_learnable_pos_embed: bool = True,
                 use_pre_norm: bool = True):
        """
        Initialize Enhanced Vision Transformer.
        
        Args:
            img_size: (H, W) image size (height=frames, width=2*joints)
            patch_size: (patch_h, patch_w) patch size
            in_channels: Number of input channels (1 for skeleton 2D image)
            embed_dim: Embedding dimension
            num_layers: Number of transformer layers (increased for robustness)
            num_heads: Number of attention heads (increased for robust multi-head attention)
            mlp_ratio: MLP hidden dimension ratio
            dropout: Dropout probability
            attention_dropout: Attention dropout probability
            num_classes: Number of action classes
            use_learnable_pos_embed: Whether to use learnable positional embeddings
            use_pre_norm: Use pre-normalization (more stable)
        """
        super(SkeletonViT, self).__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        
        # Patch embedding for 2D image
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        # Class token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Positional embedding
        if use_learnable_pos_embed:
            self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
        else:
            self.register_buffer('pos_embed', self._get_sinusoidal_pos_embed(embed_dim, num_patches + 1))
        
        # Enhanced transformer blocks with robust multi-head attention
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout, attention_dropout, use_pre_norm)
            for _ in range(num_layers)
        ])
        
        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights using standard ViT initialization."""
        # Initialize class token
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # Initialize positional embedding
        if isinstance(self.pos_embed, nn.Parameter):
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # Initialize patch embedding
        nn.init.xavier_uniform_(self.patch_embed.proj.weight)
        if self.patch_embed.proj.bias is not None:
            nn.init.zeros_(self.patch_embed.proj.bias)
        
        # Initialize transformer blocks (MultiheadAttention handles its own initialization)
        # We can apply custom initialization if needed, but default is usually fine
        
        # Initialize classification head
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        
    def _get_sinusoidal_pos_embed(self, embed_dim: int, num_positions: int) -> torch.Tensor:
        """Generate sinusoidal positional embeddings."""
        position = torch.arange(num_positions).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * 
                           -(math.log(10000.0) / embed_dim))
        pos_embed = torch.zeros(1, num_positions, embed_dim)
        pos_embed[0, :, 0::2] = torch.sin(position * div_term)
        pos_embed[0, :, 1::2] = torch.cos(position * div_term)
        return pos_embed
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape [B, 1, H, W] (2D image)
            
        Returns:
            Logits of shape [B, num_classes]
        """
        B = x.shape[0]
        
        # Patch embedding: [B, 1, H, W] -> [B, num_patches, embed_dim]
        x = self.patch_embed(x)
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, embed_dim]
        x = torch.cat([cls_tokens, x], dim=1)  # [B, num_patches + 1, embed_dim]
        
        # Add positional embedding
        x = x + self.pos_embed
        x = self.dropout(x)
        
        # Enhanced transformer blocks with robust multi-head attention
        for block in self.blocks:
            x = block(x)
        
        # Classification
        x = self.norm(x)
        cls_token_final = x[:, 0]  # [B, embed_dim]
        logits = self.head(cls_token_final)  # [B, num_classes]
        
        return logits


class GraphConvolution(nn.Module):
    """Graph Convolution layer for ST-GCN."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super(GraphConvolution, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 
                             kernel_size=(1, 1), bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [B, C, J, T]
            A: Adjacency matrix [J, J]
            
        Returns:
            Output tensor [B, C_out, J, T]
        """
        # Graph convolution: A @ x
        # x: [B, C, J, T] -> [B, C, T, J] -> [B, C, T, J] @ A^T -> [B, C, T, J] -> [B, C, J, T]
        x = x.permute(0, 1, 3, 2)  # [B, C, T, J]
        x = torch.matmul(x, A.t())  # [B, C, T, J]
        x = x.permute(0, 1, 3, 2)  # [B, C, J, T]
        
        # Convolution and batch norm
        x = self.conv(x)  # [B, C_out, J, T]
        x = self.bn(x)
        
        return x


class STGCNBlock(nn.Module):
    """Spatial-Temporal Graph Convolution Block."""
    
    def __init__(self, in_channels: int, out_channels: int,
                 stride: int = 1, residual: bool = True):
        super(STGCNBlock, self).__init__()
        self.residual = residual
        
        # Spatial graph convolution
        self.spatial_gcn = GraphConvolution(in_channels, out_channels)
        
        # Temporal convolution
        self.temporal_conv = nn.Conv2d(out_channels, out_channels, 
                                       kernel_size=(1, 9), stride=(1, stride), 
                                       padding=(0, 4), bias=False)
        self.temporal_bn = nn.BatchNorm2d(out_channels)
        
        # Residual connection
        if residual and in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, 
                                           kernel_size=1, bias=False)
            self.residual_bn = nn.BatchNorm2d(out_channels)
        elif residual:
            self.residual_conv = None
            self.residual_bn = None
        else:
            self.residual_conv = None
            self.residual_bn = None
    
    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [B, C, J, T]
            A: Adjacency matrix [J, J]
            
        Returns:
            Output tensor [B, C_out, J, T]
        """
        residual = x
        
        # Spatial graph convolution
        x = self.spatial_gcn(x, A)
        x = F.relu(x)
        
        # Temporal convolution
        x = self.temporal_conv(x)
        x = self.temporal_bn(x)
        x = F.relu(x)
        
        # Residual connection
        if self.residual:
            if self.residual_conv is not None:
                residual = self.residual_conv(residual)
                residual = self.residual_bn(residual)
            x = x + residual
        
        return x


def get_coco_adjacency_matrix(num_joints: int = 17) -> torch.Tensor:
    """
    Get COCO 17 keypoint adjacency matrix.
    
    COCO 17 keypoints:
    0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear,
    5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow,
    9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip,
    13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle
    """
    A = torch.zeros(num_joints, num_joints)
    
    # Define skeleton connections (undirected graph)
    edges = [
        # Head connections
        (0, 1), (0, 2), (1, 3), (2, 4),
        # Upper body
        (5, 6),  # shoulders
        (5, 7), (7, 9),  # left arm
        (6, 8), (8, 10),  # right arm
        (5, 11), (6, 12),  # shoulders to hips
        # Lower body
        (11, 12),  # hips
        (11, 13), (13, 15),  # left leg
        (12, 14), (14, 16),  # right leg
    ]
    
    for i, j in edges:
        A[i, j] = 1.0
        A[j, i] = 1.0  # Undirected
    
    # Self-connections
    A += torch.eye(num_joints)
    
    # Normalize
    D = torch.diag(torch.sum(A, dim=1) ** -0.5)
    A = D @ A @ D
    
    return A


class SkeletonSTGCN(nn.Module):
    """
    Spatial-Temporal Graph Convolutional Network for skeleton action recognition.
    Based on ST-GCN with improvements.
    
    Input shape: [B, C=2, J=17, T=32]
    Output shape: [B, num_classes=6]
    """
    
    def __init__(self, num_classes: int = 6, num_joints: int = 17, 
                 in_channels: int = 2, num_stages: int = 4,
                 base_channels: int = 64, dropout: float = 0.5):
        """
        Initialize ST-GCN model.
        
        Args:
            num_classes: Number of action classes
            num_joints: Number of skeleton joints (17 for COCO)
            in_channels: Input channels (2 for x, y coordinates)
            num_stages: Number of ST-GCN stages
            base_channels: Base number of channels
            dropout: Dropout probability
        """
        super(SkeletonSTGCN, self).__init__()
        
        self.num_joints = num_joints
        self.register_buffer('A', get_coco_adjacency_matrix(num_joints))
        
        # Input embedding
        self.input_embed = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=1),
            nn.BatchNorm2d(base_channels)
        )
        
        # ST-GCN blocks with proper channel progression
        self.stgcn_blocks = nn.ModuleList()
        # Channel progression: dynamically build based on num_stages
        # Pattern: base -> base -> base*2 -> base*2 -> base*4 -> base*4 -> ...
        in_ch = base_channels
        out_ch = base_channels
        
        for i in range(num_stages):
            if i == 0:
                in_ch = base_channels
                out_ch = base_channels
            elif i == 1:
                in_ch = base_channels
                out_ch = base_channels * 2
            elif i % 2 == 0:
                # Even stages: keep same output channels
                in_ch = out_ch
                out_ch = out_ch
            else:
                # Odd stages: double output channels
                in_ch = out_ch
                out_ch = out_ch * 2
            
            self.stgcn_blocks.append(
                STGCNBlock(in_ch, out_ch, stride=1, residual=True)
            )
        
        # Final output channels from last stage
        final_channels = out_ch
        
        # Global pooling and classification
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(final_channels, 512)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape [B, 2, 17, 32]
            
        Returns:
            Logits of shape [B, num_classes]
        """
        # Input embedding
        x = self.input_embed(x)  # [B, 64, 17, 32]
        
        # ST-GCN blocks
        for stgcn_block in self.stgcn_blocks:
            x = stgcn_block(x, self.A)  # [B, C, 17, T]
        
        # Global pooling
        x = self.global_pool(x)  # [B, C, 1, 1]
        x = x.view(x.size(0), -1)  # [B, C]
        
        # Classification
        x = F.relu(self.fc1(x))  # [B, 512]
        x = self.dropout(x)
        x = self.fc2(x)  # [B, num_classes]
        
        return x


class TemporalConvNet(nn.Module):
    """Temporal Convolutional Network for sequence modeling."""
    
    def __init__(self, num_inputs: int, num_channels: list, kernel_size: int = 3, dropout: float = 0.2):
        """
        Initialize TCN.
        
        Args:
            num_inputs: Input dimension
            num_channels: List of channel sizes for each layer
            kernel_size: Convolution kernel size
            dropout: Dropout probability
        """
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            layers.append(nn.Conv1d(in_channels, out_channels, kernel_size,
                                   dilation=dilation_size, padding=(kernel_size-1)*dilation_size))
            layers.append(nn.BatchNorm1d(out_channels))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [B, C, T]
            
        Returns:
            Output tensor [B, C_out, T]
        """
        x = self.network(x)
        # Remove padding
        x = x[:, :, :-self.network[0].padding[0]] if self.network[0].padding[0] > 0 else x
        return x


class SkeletonTCNTE(nn.Module):
    """
    TCN + Transformer Encoder (TCNTE) for skeleton action recognition.
    Uses TCN for temporal modeling and Transformer for attention.
    
    Input shape: [B, C=2, J=17, T=32]
    Output shape: [B, num_classes=6]
    """
    
    def __init__(self, num_classes: int = 6, num_joints: int = 17,
                 in_channels: int = 2, embed_dim: int = 128,
                 tcn_channels: list = [64, 128, 256],
                 num_transformer_layers: int = 4, num_heads: int = 8,
                 mlp_ratio: float = 4.0, dropout: float = 0.1):
        """
        Initialize TCNTE model.
        
        Args:
            num_classes: Number of action classes
            num_joints: Number of skeleton joints
            in_channels: Input channels (2 for x, y)
            embed_dim: Embedding dimension for transformer
            tcn_channels: List of TCN channel sizes
            num_transformer_layers: Number of transformer layers
            num_heads: Number of attention heads
            mlp_ratio: MLP expansion ratio
            dropout: Dropout probability
        """
        super(SkeletonTCNTE, self).__init__()
        
        # Joint embedding: project each joint to embed_dim
        self.joint_embed = nn.Linear(in_channels, embed_dim)
        
        # TCN for temporal modeling (applied per joint)
        self.tcn = TemporalConvNet(embed_dim, tcn_channels, kernel_size=3, dropout=dropout)
        
        # Project TCN output back to embed_dim
        self.tcn_proj = nn.Linear(tcn_channels[-1], embed_dim)
        
        # Positional encoding for temporal dimension
        self.pos_embed = nn.Parameter(torch.randn(1, num_joints, 32, embed_dim))
        
        # Transformer encoder layers
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout, dropout)
            for _ in range(num_transformer_layers)
        ])
        
        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(embed_dim, 512)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape [B, 2, 17, 32]
            
        Returns:
            Logits of shape [B, num_classes]
        """
        B, C, J, T = x.shape
        
        # Reshape: [B, C, J, T] -> [B, J, T, C]
        x = x.permute(0, 2, 3, 1)  # [B, J, T, C]
        
        # Joint embedding: [B, J, T, C] -> [B, J, T, embed_dim]
        x = self.joint_embed(x)  # [B, J, T, embed_dim]
        
        # TCN: apply temporal convolution per joint
        # Reshape for TCN: [B, J, T, embed_dim] -> [B*J, embed_dim, T]
        B, J, T_orig, D = x.shape
        x = x.view(B * J, T_orig, D).permute(0, 2, 1)  # [B*J, embed_dim, T]
        x = self.tcn(x)  # [B*J, tcn_channels[-1], T'] (T' might be different due to padding)
        
        # Reshape back: [B*J, C_out, T'] -> [B, J, T', C_out]
        tcn_out_dim = x.shape[1]
        T_new = x.shape[2]
        x = x.permute(0, 2, 1).view(B, J, T_new, tcn_out_dim)  # [B, J, T', tcn_out_dim]
        
        # Project to embed_dim
        x = self.tcn_proj(x)  # [B, J, T', embed_dim]
        
        # Add positional encoding (truncate or pad to match T_new)
        if T_new <= 32:
            x = x + self.pos_embed[:, :, :T_new, :]  # [B, J, T', embed_dim]
        else:
            # Pad positional encoding if needed
            pos_embed_padded = F.pad(self.pos_embed, (0, 0, 0, T_new - 32, 0, 0, 0, 0))
            x = x + pos_embed_padded[:, :, :T_new, :]
        
        # Reshape for transformer: [B, J, T', embed_dim] -> [B, J*T', embed_dim]
        B, J, T_new, D = x.shape
        x = x.view(B, J * T_new, D)  # [B, J*T', embed_dim]
        
        # Transformer encoder
        for transformer_layer in self.transformer_layers:
            x = transformer_layer(x)  # [B, J*T, embed_dim]
        
        # Normalize
        x = self.norm(x)  # [B, J*T, embed_dim]
        
        # Reshape for pooling: [B, J*T', embed_dim] -> [B, embed_dim, J, T']
        x = x.view(B, J, T_new, D).permute(0, 3, 1, 2)  # [B, embed_dim, J, T']
        
        # Global pooling
        x = self.global_pool(x)  # [B, embed_dim, 1, 1]
        x = x.view(B, -1)  # [B, embed_dim]
        
        # Classification
        x = F.relu(self.fc1(x))  # [B, 512]
        x = self.dropout(x)
        x = self.fc2(x)  # [B, num_classes]
        
        return x


def create_model(model_type: str, num_classes: int = 6, **kwargs) -> nn.Module:
    """
    Factory function to create a model.
    
    Args:
        model_type: '3dcnn_simple', '3dcnn_deep', '2dcnn_resnet', '2dcnn_lenet', 'vit', 'stgcn', or 'tcnt'
        num_classes: Number of action classes
        **kwargs: Additional model-specific arguments
        
    Returns:
        Model instance
    """
    if model_type == '3dcnn_simple':
        return Skeleton3DCNN_Simple(num_classes=num_classes, **kwargs)
    elif model_type == '3dcnn_deep':
        return Skeleton3DCNN_Deep(num_classes=num_classes, **kwargs)
    elif model_type == '2dcnn_resnet':
        return CustomResNet(num_classes=num_classes, **kwargs)
    elif model_type == '2dcnn_lenet':
        return LeNetRegularized(numChannels=1, classes=num_classes, **kwargs)
    elif model_type == '2dcnn':
        # Backward compatibility: default to ResNet
        return CustomResNet(num_classes=num_classes, **kwargs)
    elif model_type == 'vit':
        # ViT now uses 2D image input like 2D CNN
        return SkeletonViT(num_classes=num_classes, **kwargs)
    elif model_type == 'stgcn':
        return SkeletonSTGCN(num_classes=num_classes, **kwargs)
    elif model_type == 'tcnt':
        return SkeletonTCNTE(num_classes=num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown model_type: {model_type}. "
                        f"Supported: '3dcnn_simple', '3dcnn_deep', '2dcnn_resnet', '2dcnn_lenet', '2dcnn', 'vit', 'stgcn', 'tcnt'")


if __name__ == "__main__":
    # Test models
    batch_size = 4
    
    # Test 3D CNN Simple
    print("Testing 3D CNN Simple...")
    model1 = Skeleton3DCNN_Simple(num_classes=6)
    x1 = torch.randn(batch_size, 2, 17, 32)
    out1 = model1(x1)
    print(f"Input: {x1.shape}, Output: {out1.shape}")
    
    # Test 3D CNN Deep
    print("\nTesting 3D CNN Deep...")
    model2 = Skeleton3DCNN_Deep(num_classes=6)
    x2 = torch.randn(batch_size, 2, 17, 32)
    out2 = model2(x2)
    print(f"Input: {x2.shape}, Output: {out2.shape}")
    
    # Test 2D CNN ResNet
    print("\nTesting 2D CNN ResNet...")
    model3 = CustomResNet(num_classes=6)
    x3 = torch.randn(batch_size, 1, 32, 34)
    out3 = model3(x3)
    print(f"Input: {x3.shape}, Output: {out3.shape}")
    
    # Test 2D CNN LeNet
    print("\nTesting 2D CNN LeNet...")
    model4 = LeNetRegularized(numChannels=1, classes=6)
    x4 = torch.randn(batch_size, 1, 32, 34)
    out4 = model4(x4)
    print(f"Input: {x4.shape}, Output: {out4.shape}")
    
    # Test ViT
    print("\nTesting ViT...")
    model5 = SkeletonViT(num_classes=6)
    x5 = torch.randn(batch_size, 2, 17, 32)
    out5 = model5(x5)
    print(f"Input: {x5.shape}, Output: {out5.shape}")

