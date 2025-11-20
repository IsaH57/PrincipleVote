"""
Plackett-Luce Models in PyTorch.

The Plackett-Luce model defines a probability distribution over rankings.
For pairwise comparisons (i vs j), it simplifies to the Bradley-Terry model:
P(i > j) = exp(s_i) / (exp(s_i) + exp(s_j)) = sigmoid(s_i - s_j)

We implement two variants:
1. TabularPL: Learns a distinct score s_i for each item i. (Like a lookup table)
2. NeuralPL: Learns a function s(x) that maps item features x to a score. (Zero-shot capable)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class TabularPL(nn.Module):
    """
    Learns a scalar score for each item ID directly.
    Equivalent to logistic regression on one-hot encoded item IDs.
    Cannot generalize to new items (Cold Start problem).
    """
    def __init__(self, num_items):
        super().__init__()
        # Embedding dim is 1 because we just want a scalar score per item
        self.score_embedding = nn.Embedding(num_items, 1)
        # Initialize weights to 0 for neutral start
        nn.init.zeros_(self.score_embedding.weight)

    def forward(self, item_ids):
        # item_ids: (batch_size,)
        # returns: (batch_size, 1)
        return self.score_embedding(item_ids)

    def predict_proba(self, id_a, id_b):
        """Returns P(a > b)"""
        s_a = self.forward(id_a)
        s_b = self.forward(id_b)
        return torch.sigmoid(s_a - s_b)

class NeuralPL(nn.Module):
    """
    Learns to map item content features to a scalar score.
    Can generalize to new items if their features are provided.
    """
    def __init__(self, input_dim, hidden_dim=64, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, features):
        # features: (batch_size, input_dim)
        # returns: (batch_size, 1)
        return self.net(features)

    def predict_proba(self, feat_a, feat_b):
        """Returns P(a > b)"""
        s_a = self.forward(feat_a)
        s_b = self.forward(feat_b)
        return torch.sigmoid(s_a - s_b)

def pairwise_loss(score_winner, score_loser):
    """
    Negative Log Likelihood of the Bradley-Terry model.
    L = -log P(winner > loser)
      = -log( sigmoid(s_w - s_l) )
      = softplus(s_l - s_w)
    """
    return F.softplus(score_loser - score_winner).mean()
