"""
Tier 3 PyG Basics: Custom InMemoryDataset Builder for Static ASL Hand Keypoints.

Converts static sign keypoint JSON files into a torch_geometric.data.InMemoryDataset.
Constructs the undirected bone connectivity graph in Coordinate Format (COO) edge_index [2, 40].
Supports generating or parsing 10 static sign alphabet classes ('A' through 'J').
"""

import os
import json
import shutil
from pathlib import Path
import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset

# 20 standard anatomical hand bone connections
HAND_BONES = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index finger
    (0, 9), (9, 10), (10, 11), (11, 12),   # Middle finger
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring finger
    (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky finger
]

CLASSES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

def get_hand_edge_index() -> torch.Tensor:
    """
    Construct bidirectional Coordinate Format (COO) edge_index tensor:
    20 undirected bones -> 40 directed edges, shape [2, 40].
    """
    src, dst = [], []
    for u, v in HAND_BONES:
        src.extend([u, v])
        dst.extend([v, u])
    return torch.tensor([src, dst], dtype=torch.long)

def generate_sample_sign_data(raw_dir: str, num_samples_per_class: int = 30):
    """
    Generate synthetic sample keypoints for 10 static ASL alphabet classes
    with realistic pose perturbations and finger configurations.
    """
    raw_path = Path(raw_dir)
    raw_path.mkdir(parents=True, exist_ok=True)
    
    # Base canonical open palm coords
    base_coords = np.array([
        [0.50, 0.85, 0.00], [0.42, 0.76, -0.02], [0.35, 0.68, -0.04], [0.30, 0.60, -0.06], [0.26, 0.54, -0.07],
        [0.42, 0.52, -0.03], [0.39, 0.40, -0.05], [0.37, 0.30, -0.06], [0.35, 0.22, -0.07],
        [0.50, 0.50, -0.02], [0.50, 0.36, -0.04], [0.50, 0.26, -0.06], [0.50, 0.17, -0.07],
        [0.58, 0.52, -0.03], [0.60, 0.40, -0.05], [0.62, 0.31, -0.06], [0.63, 0.23, -0.07],
        [0.66, 0.56, -0.02], [0.70, 0.46, -0.04], [0.73, 0.38, -0.05], [0.75, 0.31, -0.06]
    ], dtype=np.float32)
    
    np.random.seed(42)
    for class_idx, class_name in enumerate(CLASSES):
        class_dir = raw_path / class_name
        class_dir.mkdir(exist_ok=True)
        
        for sample_i in range(num_samples_per_class):
            # Apply class-distinctive bending patterns
            coords = base_coords.copy()
            
            # Specific distinctive geometric transformations per class:
            if class_name == "A": # Closed fist, thumb alongside
                coords[6:9, 1] += 0.25
                coords[10:13, 1] += 0.25
                coords[14:17, 1] += 0.25
                coords[18:21, 1] += 0.25
            elif class_name == "B": # Open flat palm, thumb tucked
                coords[3:5, 0] += 0.15
            elif class_name == "C": # Curved C shape
                coords[6:9, 0] += 0.10
                coords[10:13, 0] += 0.05
            elif class_name == "D": # Index up, others curled
                coords[10:13, 1] += 0.25
                coords[14:17, 1] += 0.25
                coords[18:21, 1] += 0.25
            elif class_name == "I": # Pinky up, others curled
                coords[6:9, 1] += 0.25
                coords[10:13, 1] += 0.25
                coords[14:17, 1] += 0.25
            else:
                coords[6:9, 1] += 0.10 * (class_idx % 3)
                
            # Add small random noise to simulate natural signer variance
            noise = np.random.normal(0.0, 0.012, size=coords.shape).astype(np.float32)
            sample_coords = coords + noise
            
            file_data = {
                "sign_label": class_name,
                "label_id": class_idx,
                "keypoints": sample_coords.tolist()
            }
            with open(class_dir / f"sample_{sample_i:03d}.json", "w", encoding="utf-8") as f:
                json.dump(file_data, f)

class ASLHandDataset(InMemoryDataset):
    """PyG InMemoryDataset representing static ASL hand keypoints."""
    
    def __init__(self, root: str, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0])
        
    @property
    def raw_file_names(self):
        # Scan raw files across classes
        files = []
        raw_p = Path(self.raw_dir)
        if raw_p.exists():
            for c in CLASSES:
                c_dir = raw_p / c
                if c_dir.exists():
                    files.extend([str(p.relative_to(self.raw_dir)) for p in c_dir.glob("*.json")])
        return files
        
    @property
    def processed_file_names(self):
        return ["asl_hand_data.pt"]
        
    def download(self):
        # If raw directory is empty, generate structured sign data
        if not any(Path(self.raw_dir).glob("*/*.json")):
            print("Generating 10-class static ASL sample keypoint dataset...")
            generate_sample_sign_data(self.raw_dir)
            
    def process(self):
        edge_index = get_hand_edge_index()
        data_list = []
        
        raw_p = Path(self.raw_dir)
        for class_idx, class_name in enumerate(CLASSES):
            class_dir = raw_p / class_name
            if not class_dir.exists():
                continue
            for json_file in sorted(class_dir.glob("*.json")):
                with open(json_file, "r", encoding="utf-8") as f:
                    entry = json.load(f)
                    
                pts = np.array(entry["keypoints"], dtype=np.float32) # [21, 3]
                x = torch.tensor(pts, dtype=torch.float32)
                y = torch.tensor([class_idx], dtype=torch.long)
                
                data = Data(x=x, edge_index=edge_index, y=y)
                data_list.append(data)
                
        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]
            
        self.save(data_list, self.processed_paths[0])
        print(f"Processed and cached {len(data_list)} hand graph samples into {self.processed_paths[0]}")

if __name__ == "__main__":
    dataset_root = Path(__file__).resolve().parent / "asl_dataset"
    dataset = ASLHandDataset(root=str(dataset_root))
    
    print("=== TIER 3: PYG DATASET BUILDER ===")
    print(f"Total Graphs: {len(dataset)}")
    print(f"Number of Classes: {len(CLASSES)} ({CLASSES})")
    sample = dataset[0]
    print(f"Sample Graph 0: Nodes={sample.num_nodes}, Edges={sample.num_edges}, Features={sample.num_node_features}, Label={sample.y.item()} ('{CLASSES[sample.y.item()]}')")
    print(f"Edge Index shape (COO): {sample.edge_index.shape}")
