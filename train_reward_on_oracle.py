"""
Train reward model on oracle winner predictions from voting profiles.
Uses the pairwise comparison dataset generated from oracle winners.
"""

import os
import json
import math
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Dict, List
from dataclasses import dataclass

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class OraclePairwiseDataset(Dataset):
    """Dataset for oracle pairwise comparisons."""
    
    def __init__(self, csv_file: str):
        self.data = pd.read_csv(csv_file)
        print(f"Loaded {len(self.data)} pairwise comparisons from {csv_file}")
        
        # Features: plurality, borda, copeland scores for winner and loser
        self.feature_cols = [
            'winner_plurality', 'winner_borda', 'winner_copeland',
            'loser_plurality', 'loser_borda', 'loser_copeland'
        ]
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Extract features
        winner_features = torch.tensor([
            row['winner_plurality'],
            row['winner_borda'],
            row['winner_copeland']
        ], dtype=torch.float32)
        
        loser_features = torch.tensor([
            row['loser_plurality'],
            row['loser_borda'],
            row['loser_copeland']
        ], dtype=torch.float32)
        
        # Normalize by number of voters (assuming 55)
        winner_features = winner_features / 55.0
        loser_features = loser_features / 55.0
        
        return {
            'winner_feat': winner_features,
            'loser_feat': loser_features,
            'margin': torch.tensor(row['margin'], dtype=torch.float32),
            'profile_id': torch.tensor(row['profile_id'], dtype=torch.long)
        }


class RewardMLP(nn.Module):
    """MLP for scoring candidates."""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [64, 32]):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x).squeeze(-1)  # (B,)


@dataclass
class TrainConfig:
    batch_size: int = 256
    epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 1e-4
    val_split: float = 0.2
    seed: int = 42


def pairwise_loss(w_scores: torch.Tensor, l_scores: torch.Tensor) -> torch.Tensor:
    """Bradley-Terry loss: -log(sigmoid(w - l))."""
    return -F.logsigmoid(w_scores - l_scores).mean()


def compute_pairwise_metrics(model: RewardMLP, dataloader: DataLoader) -> Dict[str, float]:
    """Compute accuracy, precision, recall, F1 on pairwise predictions."""
    model.eval()
    tp = fp = tn = fn = 0
    
    with torch.no_grad():
        for batch in dataloader:
            w_feat = batch['winner_feat'].to(DEVICE)
            l_feat = batch['loser_feat'].to(DEVICE)
            w_scores = model(w_feat)
            l_scores = model(l_feat)
            
            # Original: label = 1 (winner beats loser)
            pred_pos = (w_scores > l_scores).long()
            tp += pred_pos.sum().item()
            fn += (1 - pred_pos).sum().item()
            
            # Flipped: label = 0 (loser doesn't beat winner)
            pred_flip = (l_scores > w_scores).long()
            fp += pred_flip.sum().item()
            tn += (1 - pred_flip).sum().item()
    
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn,
        'total': total
    }


def train_model(cfg: TrainConfig, data_file: str = "data/oracle_pairwise.csv"):
    """Train reward model on oracle dataset."""
    print(f"--- Training Reward Model on Oracle Data ({DEVICE}) ---")
    
    # Load dataset
    full_dataset = OraclePairwiseDataset(data_file)
    
    # Split train/val
    val_size = int(cfg.val_split * len(full_dataset))
    train_size = len(full_dataset) - val_size
    
    torch.manual_seed(cfg.seed)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, 
                              shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, 
                           shuffle=False)
    
    print(f"Train: {len(train_dataset)} pairs | Val: {len(val_dataset)} pairs")
    
    # Initialize model
    input_dim = 3  # plurality, borda, copeland
    model = RewardMLP(input_dim).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, 
                                  weight_decay=cfg.weight_decay)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs, eta_min=1e-6
    )
    
    best_val_acc = 0.0
    best_epoch = 0
    train_history = []
    
    for epoch in range(cfg.epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            w_feat = batch['winner_feat'].to(DEVICE)
            l_feat = batch['loser_feat'].to(DEVICE)
            
            w_scores = model(w_feat)
            l_scores = model(l_feat)
            
            loss = pairwise_loss(w_scores, l_scores)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                w_feat = batch['winner_feat'].to(DEVICE)
                l_feat = batch['loser_feat'].to(DEVICE)
                w_scores = model(w_feat)
                l_scores = model(l_feat)
                loss = pairwise_loss(w_scores, l_scores)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        # Metrics
        val_metrics = compute_pairwise_metrics(model, val_loader)
        
        # Log
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{cfg.epochs} | "
                  f"Train Loss: {avg_train_loss:.4f} | "
                  f"Val Loss: {avg_val_loss:.4f} | "
                  f"Val Acc: {val_metrics['accuracy']:.4f} | "
                  f"Val F1: {val_metrics['f1']:.4f}")
        
        train_history.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            **val_metrics
        })
        
        # Save best model
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            best_epoch = epoch + 1
            torch.save(model.state_dict(), 'results/oracle_reward_model_best.pt')
        
        scheduler.step()
    
    print(f"\nBest validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}")
    
    # Load best model for final evaluation
    model.load_state_dict(torch.load('results/oracle_reward_model_best.pt'))
    
    # Final metrics on train and val
    train_metrics = compute_pairwise_metrics(model, train_loader)
    val_metrics = compute_pairwise_metrics(model, val_loader)
    
    results = {
        'config': {
            'epochs': cfg.epochs,
            'batch_size': cfg.batch_size,
            'lr': cfg.lr,
            'weight_decay': cfg.weight_decay,
            'input_dim': input_dim
        },
        'best_epoch': best_epoch,
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'training_history': train_history
    }
    
    # Save results
    os.makedirs('results', exist_ok=True)
    with open('results/oracle_reward_model_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=int)
    
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print("\nTrain Metrics:")
    for k, v in train_metrics.items():
        if k in ('tp', 'fp', 'tn', 'fn', 'total'):
            print(f"  {k}: {v}")
        else:
            print(f"  {k}: {v:.4f}")
    
    print("\nValidation Metrics:")
    for k, v in val_metrics.items():
        if k in ('tp', 'fp', 'tn', 'fn', 'total'):
            print(f"  {k}: {v}")
        else:
            print(f"  {k}: {v:.4f}")
    
    print(f"\nModel saved to: results/oracle_reward_model_best.pt")
    print(f"Results saved to: results/oracle_reward_model_results.json")
    
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Train reward model on oracle data')
    parser.add_argument('--data-file', type=str, 
                       default='data/oracle_pairwise.csv',
                       help='Path to oracle pairwise dataset')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed
    )
    
    train_model(cfg, data_file=args.data_file)


if __name__ == '__main__':
    main()
