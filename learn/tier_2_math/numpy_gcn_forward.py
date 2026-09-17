"""
Tier 2 Math: Pure NumPy Graph Convolutional Network (GCN) Forward Pass.

Implements Kipf & Welling (2016) forward propagation without deep learning frameworks:
  H^(1) = ReLU( A_hat @ H^(0) @ W )

Where:
  - A_hat in R^(21 x 21): Symmetric normalized adjacency matrix from matrix_laplacian.py.
  - H^(0) in R^(21 x 3): Initial spatial feature matrix (e.g. 3D coordinates x, y, z).
  - W in R^(3 x 16): Linear projection weight matrix.
  - H^(1) in R^(21 x 16): First-layer graph-convolved joint representations.
"""

import json
from pathlib import Path
import numpy as np

from matrix_laplacian import build_hand_adjacency, compute_symmetric_normalized_laplacian, LANDMARK_NAMES

def relu(x: np.ndarray) -> np.ndarray:
    """Element-wise Rectified Linear Unit activation."""
    return np.maximum(0.0, x)

def gcn_layer_forward(A_hat: np.ndarray, H_in: np.ndarray, W: np.ndarray, bias: np.ndarray = None):
    """
    Compute single GCN layer forward pass:
      Z = A_hat @ H_in @ W (+ bias)
      H_out = ReLU(Z)
    """
    # 1. Spatial aggregation across graph neighborhood: H_agg = A_hat @ H_in
    H_agg = A_hat @ H_in
    
    # 2. Linear feature projection: Z = H_agg @ W
    Z = H_agg @ W
    if bias is not None:
        Z += bias
        
    # 3. Non-linear activation: H_out = ReLU(Z)
    H_out = relu(Z)
    
    return H_agg, Z, H_out

def run_forward_pass(seed: int = 42):
    """Execute complete forward pass with deterministic seed."""
    np.random.seed(seed)
    
    # Adjacency and symmetric normalization
    A = build_hand_adjacency()
    _, _, _, A_hat = compute_symmetric_normalized_laplacian(A)
    
    # Initial features H^(0): 21 hand keypoints in 3D spatial space
    # Simulating normalized hand coordinates in [0, 1]
    H_0 = np.random.uniform(-1.0, 1.0, size=(21, 3)).astype(np.float32)
    
    # Weight matrix W^(0) projecting 3 spatial coordinates -> 16 latent channels
    # Xavier / Glorot uniform initialization: limit = sqrt(6 / (in + out))
    limit = np.sqrt(6.0 / (3 + 16))
    W_0 = np.random.uniform(-limit, limit, size=(3, 16)).astype(np.float32)
    bias_0 = np.zeros((16,), dtype=np.float32)
    
    H_agg, Z, H_1 = gcn_layer_forward(A_hat, H_0, W_0, bias_0)
    
    return A_hat, H_0, W_0, H_agg, Z, H_1

def export_forward_results(A_hat, H_0, W_0, H_agg, Z, H_1, out_path: str):
    """Export matrix calculation steps to JSON for UI step-by-step inspector."""
    data = {
        "nodes": LANDMARK_NAMES,
        "shapes": {
            "A_hat": list(A_hat.shape),
            "H_0": list(H_0.shape),
            "W_0": list(W_0.shape),
            "H_agg": list(H_agg.shape),
            "Z": list(Z.shape),
            "H_1": list(H_1.shape)
        },
        "H_0": np.round(H_0, 4).tolist(),
        "W_0": np.round(W_0, 4).tolist(),
        "H_agg": np.round(H_agg, 4).tolist(),
        "H_1": np.round(H_1, 4).tolist(),
        "sparsity_ratio": float(np.mean(H_1 == 0.0))
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

if __name__ == "__main__":
    A_hat, H_0, W_0, H_agg, Z, H_1 = run_forward_pass()
    
    print("=== TIER 2: PURE NUMPY GCN FORWARD PASS ===")
    print(f"H^(0) (Spatial Features) Shape:      {H_0.shape}")
    print(f"A_hat (Normalized Adjacency) Shape: {A_hat.shape}")
    print(f"W^(0) (Projection Weight) Shape:    {W_0.shape}")
    print(f"H_agg = A_hat @ H^(0) Shape:        {H_agg.shape}")
    print(f"H^(1) = ReLU(H_agg @ W^(0)) Shape:  {H_1.shape}")
    print(f"ReLU Zero-Activation Sparsity:      {np.mean(H_1 == 0.0) * 100:.1f}%")
    print(f"Sample Node 0 (Wrist) H^(1)[:5]:    {np.round(H_1[0, :5], 4)}")
    
    export_path = Path(__file__).resolve().parent.parent / "web" / "data" / "numpy_gcn_forward.json"
    export_forward_results(A_hat, H_0, W_0, H_agg, Z, H_1, str(export_path))
    print(f"Exported forward pass data to: {export_path}")
