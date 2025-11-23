"""
Train a lightweight reward model on synthetic pairwise restaurant comparisons
and analyze which social choice aggregation it approximates plus basic axiom
 satisfaction metrics.

Pipeline:
1. Load restaurant + persona data (embeddings + utility matrix).
2. Generate pairwise comparison dataset (winner/loser embeddings).
3. Train an MLP reward model with Bradley-Terry logistic loss: -log(sigmoid(r_w - r_l)).
4. Score all restaurants with trained reward model.
5. Compare resulting ranking against standard aggregation methods:
   - Utilitarian (mean utility)
   - Plurality (top-1 counts across personas)
   - Borda (per-persona position scores)
   - Copeland (pairwise majority wins minus losses)
6. Axiom checks (fraction satisfied): Pareto dominance, Condorcet (if winner exists),
   and Anonymity proxy (robustness under persona permutation sampling).
7. Save model + JSON summary in results/.
"""

import os
import json
import math
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import RestaurantData, PairwiseComparisonDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class RewardMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int] = [256, 128]):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)  # (B,)


@dataclass
class TrainConfig:
    batch_size: int = 512
    epochs: int = 100
    lr: float = 1e-4
    weight_decay: float = 0.0
    num_samples: int = 20000  # pairwise comparisons
    beta: float = 10.0  # for Bradley-Terry sampling in dataset
    seed: int = 42


def pairwise_loss(w_scores: torch.Tensor, l_scores: torch.Tensor) -> torch.Tensor:
    # -log(sigmoid(w - l))
    return -F.logsigmoid(w_scores - l_scores).mean()


def compute_plurality_counts(utilities: np.ndarray) -> np.ndarray:
    # Count how often each restaurant is top for a persona
    top_indices = utilities.argmax(axis=1)
    counts = np.bincount(top_indices, minlength=utilities.shape[1])
    return counts.astype(float)


def compute_borda_scores(utilities: np.ndarray) -> np.ndarray:
    # For each persona, ranking descending by utility; assign points (n-1 .. 0)
    num_personas, num_restaurants = utilities.shape
    scores = np.zeros(num_restaurants, dtype=float)
    for p in range(num_personas):
        order = np.argsort(-utilities[p])  # descending
        for rank, r_id in enumerate(order):
            scores[r_id] += num_restaurants - rank - 1
    return scores / num_personas  # normalize


def compute_copeland_scores(utilities: np.ndarray) -> np.ndarray:
    # Majority comparison per persona: restaurant i preferred over j if utility_i > utility_j
    num_personas, num_restaurants = utilities.shape
    wins = np.zeros(num_restaurants, dtype=float)
    losses = np.zeros(num_restaurants, dtype=float)
    for i in range(num_restaurants):
        for j in range(i + 1, num_restaurants):
            # Count personas preferring i over j
            pref_i = np.sum(utilities[:, i] > utilities[:, j])
            pref_j = np.sum(utilities[:, j] > utilities[:, i])
            if pref_i == pref_j:
                continue  # tie -> no update
            if pref_i > pref_j:
                wins[i] += 1
                losses[j] += 1
            else:
                wins[j] += 1
                losses[i] += 1
    return wins - losses  # Copeland score


def condorcet_winner(utilities: np.ndarray) -> int:
    num_personas, num_restaurants = utilities.shape
    for i in range(num_restaurants):
        wins_all = True
        for j in range(num_restaurants):
            if i == j:
                continue
            pref_i = np.sum(utilities[:, i] > utilities[:, j])
            pref_j = np.sum(utilities[:, j] > utilities[:, i])
            if pref_i <= pref_j:  # not strict majority
                wins_all = False
                break
        if wins_all:
            return i
    return -1


def pareto_satisfaction(utilities: np.ndarray, scores: np.ndarray) -> float:
    # Fraction of pairs where if all personas prefer i over j then score[i] > score[j]
    num_personas, num_restaurants = utilities.shape
    total = 0
    satisfied = 0
    for i in range(num_restaurants):
        for j in range(num_restaurants):
            if i == j:
                continue
            if np.all(utilities[:, i] >= utilities[:, j]) and np.any(utilities[:, i] > utilities[:, j]):
                total += 1
                if scores[i] > scores[j]:
                    satisfied += 1
    return satisfied / total if total > 0 else 1.0


def anonymity_proxy(restaurant_embeddings: np.ndarray, model: RewardMLP, utilities: np.ndarray, trials: int = 5) -> float:
    # Shuffle personas (utilities rows) and recompute aggregation baseline; since model sees only restaurant embeddings
    # its scores should be invariant -> correlation ~ 1.0
    with torch.no_grad():
        emb_tensor = torch.tensor(restaurant_embeddings, dtype=torch.float32, device=DEVICE)
        scores = model(emb_tensor).cpu().numpy()
    corrs = []
    for _ in range(trials):
        shuffled = utilities.copy()
        np.random.shuffle(shuffled)  # permute rows
        # Model scores unchanged; utilitarian baseline changes slightly -> compute correlation of model scores with new mean
        mean_util = shuffled.mean(axis=0)
        rho = spearman_corr(scores, mean_util)
        corrs.append(rho)
    return float(np.mean(corrs))


def spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr
    rho, _ = spearmanr(a, b)
    if math.isnan(rho):
        return 0.0
    return float(rho)


def compute_pairwise_metrics(model: RewardMLP, dataset: PairwiseComparisonDataset, batch_size: int = 1024) -> Dict[str, float]:
    """Compute accuracy / precision / recall / F1 for pairwise winner prediction.
    We construct a balanced evaluation set by adding a flipped (loser,winner) example
    for each original pair so negatives are present.
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    tp = fp = tn = fn = 0
    with torch.no_grad():
        for batch in loader:
            w_feat = batch['winner_feat'].to(DEVICE)
            l_feat = batch['loser_feat'].to(DEVICE)
            w_scores = model(w_feat)
            l_scores = model(l_feat)
            # Original orientation: label = 1 (winner)
            pred_pos = (w_scores > l_scores).long()
            tp += pred_pos.sum().item()
            fn += (1 - pred_pos).sum().item()
            # Flipped orientation: label = 0 (loser now in winner position)
            pred_flip = (l_scores > w_scores).long()  # predicts loser beats winner
            fp += pred_flip.sum().item()  # predicting positive when label is 0
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
        'num_pairs': len(dataset)
    }


def train_reward_model(cfg: TrainConfig) -> Dict:
    print(f"--- Reward Model Training ({DEVICE}) ---")
    r_data = RestaurantData(restaurants_file="restaurants.csv")
    dataset = PairwiseComparisonDataset(r_data, num_samples=cfg.num_samples, beta=cfg.beta, seed=cfg.seed)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True)

    input_dim = r_data.restaurant_embeddings.shape[1]
    model = RewardMLP(input_dim).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    for epoch in range(cfg.epochs):
        model.train()
        total_loss = 0.0
        for batch in loader:
            w_feat = batch['winner_feat'].to(DEVICE)
            l_feat = batch['loser_feat'].to(DEVICE)
            w_scores = model(w_feat)
            l_scores = model(l_feat)
            loss = pairwise_loss(w_scores, l_scores)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}/{cfg.epochs} | Loss: {avg_loss:.4f}")

    # Scoring all restaurants
    model.eval()
    with torch.no_grad():
        all_emb = torch.tensor(r_data.restaurant_embeddings, dtype=torch.float32, device=DEVICE)
        reward_scores = model(all_emb).cpu().numpy()

    utilities = r_data.utilities  # personas x restaurants
    mean_util = utilities.mean(axis=0)
    plurality = compute_plurality_counts(utilities)
    borda = compute_borda_scores(utilities)
    copeland = compute_copeland_scores(utilities)

    correlations = {
        'utilitarian_mean': spearman_corr(reward_scores, mean_util),
        'plurality': spearman_corr(reward_scores, plurality),
        'borda': spearman_corr(reward_scores, borda),
        'copeland': spearman_corr(reward_scores, copeland),
    }

    # Classification metrics on pairwise dataset
    pairwise_metrics = compute_pairwise_metrics(model, dataset)

    pareto_ratio = pareto_satisfaction(utilities, reward_scores)
    condorcet_idx = condorcet_winner(utilities)
    condorcet_satisfied = False
    if condorcet_idx >= 0:
        condorcet_satisfied = int(np.argmax(reward_scores) == condorcet_idx)

    anonymity_corr = anonymity_proxy(r_data.restaurant_embeddings, model, utilities)

    results = {
        'training': {
            'epochs': cfg.epochs,
            'final_loss': avg_loss,
            'num_pairs': cfg.num_samples,
            'embedding_dim': input_dim
        },
        'correlations': correlations,
        'pairwise_metrics': pairwise_metrics,
        'axioms': {
            'pareto_fraction': pareto_ratio,
            'condorcet_winner_exists': condorcet_idx >= 0,
            'condorcet_ranked_first': condorcet_satisfied,
            'anonymity_proxy_mean_rho': anonymity_corr
        }
    }

    os.makedirs('results', exist_ok=True)
    torch.save(model.state_dict(), 'results/reward_model.pt')
    with open('results/reward_model_analysis.json', 'w') as f:
        json.dump(results, f, indent=2)
    print('Saved model to results/reward_model.pt and analysis to results/reward_model_analysis.json')

    # Print summary table
    print('\nCorrelation with aggregation methods:')
    for k, v in correlations.items():
        print(f"  {k}: {v:.3f}")
    print('\nAxiom satisfaction:')
    for k, v in results['axioms'].items():
        print(f"  {k}: {v}")
    print('\nPairwise prediction metrics:')
    for k, v in results['pairwise_metrics'].items():
        if k in ('tp','fp','tn','fn','num_pairs'):
            print(f"  {k}: {v}")
        else:
            print(f"  {k}: {v:.4f}")

    return results


def main():
    cfg = TrainConfig()
    train_reward_model(cfg)


if __name__ == '__main__':
    main()
