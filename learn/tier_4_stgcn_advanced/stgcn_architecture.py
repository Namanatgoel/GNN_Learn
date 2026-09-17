"""
Tier 4 ST-GCN Advanced: Spatiotemporal Graph Convolutional Network Architecture.

Implements Yan et al. (ST-GCN) spatiotemporal reasoning for continuous sign gloss sequences:
  - Intra-frame spatial graph convolution across 21 anatomical hand bones.
  - Inter-frame temporal convolution across time steps T connecting joint v_t with v_(t+1).
  - Residual connections and batch normalization.
Hardware:
  - Fully optimized for execution on cuda:0 (NVIDIA GeForce RTX 5050).
"""

import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 20 standard hand anatomical bone connections
HAND_BONES = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index finger
    (0, 9), (9, 10), (10, 11), (11, 12),   # Middle finger
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring finger
    (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky finger
]

def build_normalized_spatial_adjacency(num_nodes: int = 21) -> torch.Tensor:
    """Build Kipf-Welling normalized spatial adjacency matrix A_hat in R^(V x V)."""
    A = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for u, v in HAND_BONES:
        A[u, v] = 1.0
        A[v, u] = 1.0
    A_tilde = A + np.eye(num_nodes, dtype=np.float32)
    d_tilde = np.sum(A_tilde, axis=1)
    d_inv_sqrt = np.power(d_tilde, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    D_inv = np.diag(d_inv_sqrt)
    A_hat = D_inv @ A_tilde @ D_inv
    return torch.from_numpy(A_hat).float()

class SpatialGraphConv(nn.Module):
    """
    Intra-frame spatial skeletal convolution across V joints.
    Multiplies spatial adjacency A_hat with node features across each time slice.
    """
    def __init__(self, in_channels: int, out_channels: int, A_hat: torch.Tensor):
        super().__init__()
        self.register_buffer("A_hat", A_hat)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x: [B, C_in, T, V]
        # Multiply along spatial dimension V using einstein summation
        # x_agg: [B, C_in, T, V] = sum_u (A_hat[v, u] * x[B, C_in, T, u])
        x_agg = torch.einsum("vu,bctu->bctv", self.A_hat, x)
        return self.conv(x_agg)

class TemporalConv(nn.Module):
    """
    Inter-frame temporal convolution connecting joint v at time t to time t+1.
    Uses 1D/2D convolution along temporal dimension with kernel size (kernel_size, 1).
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 9, stride: int = 1):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1),
                              stride=(stride, 1), padding=(padding, 0))
        self.bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.conv(x))

class STGCNBlock(nn.Module):
    """
    Spatiotemporal Graph Convolutional Network (ST-GCN) unit:
      1. Spatial Graph Conv (intra-frame bones)
      2. Spatial BatchNorm & ReLU
      3. Temporal Conv (inter-frame joint trajectories)
      4. Residual Skip Connection
      5. ReLU Activation
    """
    def __init__(self, in_channels: int, out_channels: int, A_hat: torch.Tensor,
                 temporal_kernel_size: int = 9, stride: int = 1, dropout: float = 0.1):
        super().__init__()
        self.spatial_conv = SpatialGraphConv(in_channels, out_channels, A_hat)
        self.bn_spatial = nn.BatchNorm2d(out_channels)
        
        self.temporal_conv = TemporalConv(out_channels, out_channels,
                                          kernel_size=temporal_kernel_size, stride=stride)
        self.dropout = nn.Dropout(p=dropout)
        
        # Residual connection if channels change or stride > 1
        if in_channels != out_channels or stride != 1:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.residual = nn.Identity()
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.residual(x)
        
        # Intra-frame spatial convolution
        x = self.spatial_conv(x)
        x = self.bn_spatial(x)
        x = F.relu(x)
        
        # Inter-frame temporal convolution
        x = self.temporal_conv(x)
        x = self.dropout(x)
        
        # Add residual skip and non-linear activation
        return F.relu(x + res)

class STGCN(nn.Module):
    """Complete Spatiotemporal GCN model for dynamic sign recognition."""
    def __init__(self, in_channels: int = 3, num_classes: int = 8, num_nodes: int = 21,
                 num_frames: int = 30, hidden_dim: int = 64):
        super().__init__()
        A_hat = build_normalized_spatial_adjacency(num_nodes)
        
        self.data_bn = nn.BatchNorm1d(in_channels * num_nodes)
        
        self.block1 = STGCNBlock(in_channels, hidden_dim, A_hat, stride=1)
        self.block2 = STGCNBlock(hidden_dim, hidden_dim * 2, A_hat, stride=2)
        self.block3 = STGCNBlock(hidden_dim * 2, hidden_dim * 2, A_hat, stride=1)
        
        # Global spatiotemporal pooling over time T and nodes V
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x: [B, C, T, V]
        B, C, T, V = x.size()
        
        # Input batch normalization
        x_perm = x.permute(0, 1, 3, 2).contiguous().view(B, C * V, T)
        x_norm = self.data_bn(x_perm)
        x = x_norm.view(B, C, V, T).permute(0, 1, 3, 2).contiguous()
        
        # ST-GCN feature extraction
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        
        # Global spatiotemporal pooling: mean across T and V
        x_pooled = F.avg_pool2d(x, (x.size(2), x.size(3))).view(B, -1)
        
        # Logits output
        logits = self.classifier(x_pooled)
        return logits

def export_architecture_metadata(out_path: str):
    """Save architecture parameters to JSON for UI inspection."""
    data = {
        "model_name": "ST-GCN (Spatiotemporal Graph Convolutional Network)",
        "input_tensor_format": "[Batch, Channels=3, Frames=30, Joints=21]",
        "spatial_edges": HAND_BONES,
        "temporal_connectivity": "Joint v at frame t connected to joint v at frame t+1 (kernel_size=9)",
        "blocks": [
            {"layer": 1, "in_channels": 3, "out_channels": 64, "temporal_stride": 1},
            {"layer": 2, "in_channels": 64, "out_channels": 128, "temporal_stride": 2},
            {"layer": 3, "in_channels": 128, "out_channels": 128, "temporal_stride": 1}
        ],
        "pooling": "Global Spatiotemporal Average Pooling (T x V)",
        "device_target": "cuda:0 (NVIDIA GeForce RTX 5050)"
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("=== TIER 4: ST-GCN ARCHITECTURE VALIDATION ===")
    print(f"Instantiating model on device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    
    model = STGCN(in_channels=3, num_classes=8, num_nodes=21, num_frames=30, hidden_dim=64).to(device)
    
    # Test forward pass with synthetic batch of 30-frame dynamic sequences
    batch_size = 4
    x_sample = torch.randn(batch_size, 3, 30, 21, device=device)
    
    with torch.no_grad():
        out = model(x_sample)
        
    print(f"Input Sequence Shape:  {tuple(x_sample.shape)} -> (B, C, T, V)")
    print(f"Output Logits Shape:   {tuple(out.shape)} -> (B, NumClasses)")
    vram_mb = torch.cuda.memory_allocated(device) / 1024**2 if torch.cuda.is_available() else 0.0
    print(f"RTX 5050 Memory Usage: {vram_mb:.2f} MB")
    
    export_path = Path(__file__).resolve().parent.parent / "web" / "data" / "stgcn_architecture.json"
    export_architecture_metadata(str(export_path))
    print(f"Exported architecture metadata to: {export_path}")
