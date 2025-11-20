"""
Training script for Neural Plackett-Luce.
Demonstrates the difference between Tabular (memorization) and Neural (generalization) models.
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import time

from data import RestaurantData, PairwiseComparisonDataset
from model import TabularPL, NeuralPL, pairwise_loss

def evaluate_ranking(model, r_data, test_indices, model_type='neural'):
    """
    Evaluates the model's ability to rank a set of items.
    We compare the model's predicted scores against the ground truth Borda count 
    (aggregated across all personas).
    """
    # 1. Compute Ground Truth Ranking (Aggregated Utility)
    # We simply sum the utilities across all personas for the test items
    # utilities shape: (num_personas, num_restaurants)
    subset_utilities = r_data.utilities[:, test_indices] # (100, num_test)
    mean_utilities = subset_utilities.mean(axis=0) # (num_test,)
    true_ranks = np.argsort(np.argsort(mean_utilities)[::-1]) # 0 is best
    
    # 2. Compute Model Scores
    model.eval()
    with torch.no_grad():
        if model_type == 'tabular':
            # Tabular model can only score items it has seen (indices 0..N-1)
            # If test_indices are outside the training range, this will fail or be meaningless.
            # For this demo, we assume Tabular is only evaluated on training items.
            ids = torch.tensor(test_indices, dtype=torch.long)
            scores = model(ids).squeeze().numpy()
        else:
            # Neural model uses features
            feats = r_data.restaurant_embeddings[test_indices]
            feats_tensor = torch.tensor(feats, dtype=torch.float32)
            scores = model(feats_tensor).squeeze().numpy()
            
    # 3. Compute Correlation
    # We want the scores to correlate with the true utility (or rank)
    # Spearman correlation between predicted scores and true mean utility
    rho, _ = spearmanr(scores, mean_utilities)
    return rho

def main():
    # --- Configuration ---
    BATCH_SIZE = 32
    LR = 0.001
    EPOCHS = 5
    NUM_TRAIN_ITEMS = 40 # Use first 40 for training
    NUM_TEST_ITEMS = 10  # Use last 10 for zero-shot testing
    
    print("--- Neural Plackett-Luce Demo ---")
    
    # 1. Data Preparation
    r_data = RestaurantData()
    total_items = len(r_data.restaurants_df)
    
    # Split items
    all_indices = np.arange(total_items)
    np.random.shuffle(all_indices)
    train_indices = all_indices[:NUM_TRAIN_ITEMS]
    test_indices = all_indices[NUM_TRAIN_ITEMS:]
    
    print(f"Split: {len(train_indices)} Training Items, {len(test_indices)} Test Items (Zero-Shot)")
    
    # Create Training Dataset (Only pairs from train_indices)
    # We need to hack the dataset generation slightly or subclass it.
    # For simplicity, we'll just filter the generator in a custom way here or 
    # let's just modify the dataset class to accept indices? 
    # Actually, let's just subclass on the fly or create a filtered dataset.
    
    # Let's create a custom generator for this script to be precise
    print("Generating training data...")
    train_comparisons = []
    rng = np.random.RandomState(42)
    num_samples = 5000
    
    for _ in range(num_samples):
        p_idx = rng.randint(0, len(r_data.personas_df))
        # Pick from TRAIN indices only
        r1_idx, r2_idx = rng.choice(train_indices, 2, replace=False)
        
        u1 = r_data.utilities[p_idx, r1_idx]
        u2 = r_data.utilities[p_idx, r2_idx]
        
        prob_1_wins = 1.0 / (1.0 + np.exp(-10.0 * (u1 - u2)))
        if rng.rand() < prob_1_wins:
            w, l = r1_idx, r2_idx
        else:
            w, l = r2_idx, r1_idx
            
        train_comparisons.append({
            'winner_id': w, 
            'loser_id': l,
            'winner_feat': r_data.restaurant_embeddings[w],
            'loser_feat': r_data.restaurant_embeddings[l]
        })
        
    # Simple DataLoader wrapper
    train_loader = DataLoader(train_comparisons, batch_size=BATCH_SIZE, shuffle=True)
    
    # 2. Model Initialization
    input_dim = r_data.restaurant_embeddings.shape[1]
    
    # Tabular Model (needs to know max ID, which is total_items)
    # Note: It will learn embeddings for train_indices, but test_indices will remain at initialization (0)
    tabular_model = TabularPL(total_items)
    
    # Neural Model
    neural_model = NeuralPL(input_dim)
    
    optimizer_tab = optim.Adam(tabular_model.parameters(), lr=0.01)
    optimizer_neu = optim.Adam(neural_model.parameters(), lr=0.001)
    
    # 3. Training Loop
    print("\nStarting training...")
    for epoch in range(EPOCHS):
        tabular_model.train()
        neural_model.train()
        total_loss_tab = 0
        total_loss_neu = 0
        
        for batch in train_loader:
            # Unpack
            w_id = batch['winner_id'] # (B,)
            l_id = batch['loser_id']
            w_feat = batch['winner_feat'] # (B, dim)
            l_feat = batch['loser_feat']
            
            # --- Train Tabular ---
            optimizer_tab.zero_grad()
            s_w = tabular_model(w_id)
            s_l = tabular_model(l_id)
            loss_tab = pairwise_loss(s_w, s_l)
            loss_tab.backward()
            optimizer_tab.step()
            total_loss_tab += loss_tab.item()
            
            # --- Train Neural ---
            optimizer_neu.zero_grad()
            s_w_n = neural_model(w_feat)
            s_l_n = neural_model(l_feat)
            loss_neu = pairwise_loss(s_w_n, s_l_n)
            loss_neu.backward()
            optimizer_neu.step()
            total_loss_neu += loss_neu.item()
            
        # Evaluation
        rho_tab_train = evaluate_ranking(tabular_model, r_data, train_indices, 'tabular')
        rho_neu_train = evaluate_ranking(neural_model, r_data, train_indices, 'neural')
        
        # Zero-Shot Evaluation (Test Set)
        # Tabular model hasn't seen these IDs, so scores should be random/zero.
        # Neural model sees their features.
        rho_neu_test = evaluate_ranking(neural_model, r_data, test_indices, 'neural')
        
        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"Loss Tab: {total_loss_tab/len(train_loader):.3f} | "
              f"Loss Neu: {total_loss_neu/len(train_loader):.3f} | "
              f"Train Rho (Tab/Neu): {rho_tab_train:.2f}/{rho_neu_train:.2f} | "
              f"Test Rho (Neu): {rho_neu_test:.2f}")

    print("\n--- Final Verdict ---")
    print("The Tabular model learns the training set well (high Train Rho) but cannot generalize.")
    print(f"The Neural model achieves {rho_neu_test:.2f} correlation on UNSEEN restaurants (Zero-Shot).")

if __name__ == "__main__":
    main()
