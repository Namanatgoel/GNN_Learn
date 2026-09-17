"""
Tier 3 PyG Basics: Static Hand Graph GCN Classifier.

Architecture:
  - Input: Node coordinates [21, 3] + COO edge_index [2, 40]
  - GCNConv(3 -> 64) + ReLU + BatchNorm
  - GCNConv(64 -> 64) + ReLU + BatchNorm
  - Global Mean Pooling: [Batch * 21, 64] -> [Batch, 64]
  - Linear(64 -> 10) for 10 static sign alphabet classes
Hardware:
  - Explicitly targets cuda:0 (NVIDIA GeForce RTX 5050) with pinned memory and GPU tensor caching.
"""

import json
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader

from pyg_dataset_builder import ASLHandDataset, CLASSES

class StaticGCNClassifier(nn.Module):
    """2-layer GCN with global mean pooling for graph-level sign recognition."""
    def __init__(self, in_channels: int = 3, hidden_channels: int = 64, num_classes: int = 10, dropout: float = 0.1):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(hidden_channels, num_classes)
        
    def forward(self, x, edge_index, batch):
        # 1. First spatial graph convolution
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # 2. Second spatial graph convolution (aggregates 2-hop information)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        
        # 3. Global graph pooling across all 21 joints in each hand
        pooled = global_mean_pool(x, batch) # Shape: [batch_size, hidden_channels]
        
        # 4. Final linear classification layer
        logits = self.classifier(pooled)
        return logits

def train_model(epochs: int = 10, batch_size: int = 16, lr: float = 0.01):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Executing Static GCN Classifier on Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    
    dataset_root = Path(__file__).resolve().parent / "asl_dataset"
    dataset = ASLHandDataset(root=str(dataset_root))
    
    # Deterministic split: 80% train, 20% test
    torch.manual_seed(42)
    dataset = dataset.shuffle()
    split_idx = int(0.8 * len(dataset))
    train_dataset = dataset[:split_idx]
    test_dataset = dataset[split_idx:]
    
    # Use pin_memory=True for fast DMA transfer to RTX 5050
    pin_mem = torch.cuda.is_available()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_mem)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_mem)
    
    model = StaticGCNClassifier(in_channels=3, hidden_channels=64, num_classes=len(CLASSES)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    training_history = []
    
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for batch in train_loader:
            batch = batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * batch.num_graphs
            preds = out.argmax(dim=1)
            correct_train += (preds == batch.y).sum().item()
            total_train += batch.num_graphs
            
        train_loss = total_loss / total_train
        train_acc = correct_train / total_train
        
        # Test evaluation
        model.eval()
        correct_test = 0
        total_test = 0
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device, non_blocking=True)
                out = model(batch.x, batch.edge_index, batch.batch)
                preds = out.argmax(dim=1)
                correct_test += (preds == batch.y).sum().item()
                total_test += batch.num_graphs
        test_acc = correct_test / total_test
        
        vram_mb = torch.cuda.memory_allocated(device) / 1024**2 if torch.cuda.is_available() else 0.0
        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.1f}% | Test Acc: {test_acc*100:.1f}% | VRAM: {vram_mb:.1f} MB")
        
        training_history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc * 100, 2),
            "test_acc": round(test_acc * 100, 2),
            "vram_mb": round(vram_mb, 2)
        })
        
    duration = round(time.time() - start_time, 2)
    print(f"Training completed in {duration}s on {device}.")
    
    results = {
        "device": str(device),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "total_samples": len(dataset),
        "train_samples": len(train_dataset),
        "test_samples": len(test_dataset),
        "epochs": epochs,
        "final_test_accuracy": round(test_acc * 100, 2),
        "duration_seconds": duration,
        "history": training_history,
        "classes": CLASSES
    }
    
    # Export results for UI dashboard
    web_res_path = Path(__file__).resolve().parent.parent / "web" / "data" / "static_gcn_results.json"
    web_res_path.parent.mkdir(parents=True, exist_ok=True)
    with open(web_res_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Exported static GCN results to: {web_res_path}")
    
    return model, results

if __name__ == "__main__":
    train_model(epochs=10, batch_size=16)
