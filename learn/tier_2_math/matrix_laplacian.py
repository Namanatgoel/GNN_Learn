"""
Tier 2 Math: Matrix Laplacian and Symmetric Normalization for Hand Skeleton Graph.

Computes:
  1. Hand skeleton adjacency matrix A (21 x 21) based on anatomical bone connections.
  2. Renormalization with self-loops: A_tilde = A + I_21.
  3. Degree matrix D_tilde = diag(sum_j A_tilde_ij).
  4. Symmetric normalized adjacency: A_hat = D_tilde^(-1/2) * A_tilde * D_tilde^(-1/2).
All operations are strictly implemented via NumPy.
"""

import json
from pathlib import Path
import numpy as np

# 20 standard anatomical hand bone connections (edges)
HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index finger
    (0, 9), (9, 10), (10, 11), (11, 12),   # Middle finger
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring finger
    (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky finger
]

LANDMARK_NAMES = [
    "WRIST",
    "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP",
    "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP",
    "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP"
]

def build_hand_adjacency() -> np.ndarray:
    """Build binary undirected adjacency matrix A for the 21 hand joints."""
    A = np.zeros((21, 21), dtype=np.float32)
    for u, v in HAND_EDGES:
        A[u, v] = 1.0
        A[v, u] = 1.0
    return A

def compute_symmetric_normalized_laplacian(A: np.ndarray):
    """
    Execute Kipf-Welling renormalization trick strictly using NumPy:
      A_tilde = A + I_N
      D_tilde = diag(sum(A_tilde, axis=1))
      A_hat = D_tilde^(-1/2) @ A_tilde @ D_tilde^(-1/2)
    """
    N = A.shape[0]
    # Add self-loops (each joint retains its own state)
    A_tilde = A + np.eye(N, dtype=np.float32)
    
    # Compute degree vector d_tilde_i = sum_j A_tilde_ij
    d_tilde = np.sum(A_tilde, axis=1)
    
    # Inverted square root: d_tilde^(-1/2)
    d_inv_sqrt = np.power(d_tilde, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    
    # D_tilde^(-1/2) as diagonal matrix
    D_tilde_inv_sqrt = np.diag(d_inv_sqrt)
    
    # Symmetric normalization: D^(-1/2) @ A_tilde @ D^(-1/2)
    A_hat = D_tilde_inv_sqrt @ A_tilde @ D_tilde_inv_sqrt
    
    return A, A_tilde, d_tilde, A_hat

def export_matrix_data(A: np.ndarray, A_tilde: np.ndarray, d_tilde: np.ndarray, A_hat: np.ndarray, out_path: str):
    """Save matrix derivation results to JSON for UI inspection."""
    data = {
        "num_nodes": 21,
        "landmark_names": LANDMARK_NAMES,
        "edges": HAND_EDGES,
        "degree_vector": d_tilde.tolist(),
        "A": A.tolist(),
        "A_tilde": A_tilde.tolist(),
        "A_hat": np.round(A_hat, 4).tolist()
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

if __name__ == "__main__":
    A = build_hand_adjacency()
    A, A_tilde, d_tilde, A_hat = compute_symmetric_normalized_laplacian(A)
    
    print("=== TIER 2: MATRIX LAPLACIAN & SYMMETRIC NORMALIZATION ===")
    print(f"Adjacency A shape: {A.shape} | Total edges: {int(np.sum(A) / 2)}")
    print(f"A_tilde shape: {A_tilde.shape} (with self-loops)")
    print(f"Degrees with self-loops (min/max/wrist): {d_tilde.min():.1f} / {d_tilde.max():.1f} / Node 0: {d_tilde[0]:.1f}")
    print(f"A_hat shape: {A_hat.shape} | Normalized row 0 sum: {np.sum(A_hat[0]):.4f}")
    
    export_path = Path(__file__).resolve().parent.parent / "web" / "data" / "matrix_laplacian.json"
    export_matrix_data(A, A_tilde, d_tilde, A_hat, str(export_path))
    print(f"Exported matrix data to: {export_path}")
