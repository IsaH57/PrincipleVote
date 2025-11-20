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

def evaluate_ranking(model, r_data, model_type='neural'):
    """
    Evaluates the model's ability to rank the items in r_data.
    We compare the model's predicted scores against the ground truth Borda count 
    (aggregated across all personas).
    """
    # 1. Compute Ground Truth Ranking (Aggregated Utility)
    # utilities shape: (num_personas, num_restaurants)
    mean_utilities = r_data.utilities.mean(axis=0) # (num_restaurants,)
    num_items = len(mean_utilities)
    
    # 2. Compute Model Scores
    model.eval()
    with torch.no_grad():
        if model_type == 'tabular':
            # Check if model has embeddings for these items
            if num_items != model.score_embedding.num_embeddings:
                # Dimension mismatch implies these are new/unseen items
                return 0.0
                
            ids = torch.arange(num_items, dtype=torch.long)
            scores = model(ids).squeeze().numpy()
        else:
            # Neural model uses features
            feats = r_data.restaurant_embeddings
            feats_tensor = torch.tensor(feats, dtype=torch.float32)
            scores = model(feats_tensor).squeeze().numpy()
            
    # 3. Compute Correlation
    rho, _ = spearmanr(scores, mean_utilities)
    return rho

def main():
    # --- Configuration ---
    BATCH_SIZE = 32
    LR = 0.001
    EPOCHS = 5
    
    print("--- Neural Plackett-Luce Demo ---")
    
    # 1. Data Preparation
    print("Loading Training Data (2024)...")
    train_data = RestaurantData(restaurants_file="restaurants.csv")
    
    print("Loading Test Data (2023)...")
    test_data_2023 = RestaurantData(restaurants_file="restaurants_2023.csv")
    
    print("Loading Test Data (2025)...")
    test_data_2025 = RestaurantData(restaurants_file="restaurants_2025.csv")
    
    # Create Training Dataset
    print("Generating training pairs...")
    # We use all items in train_data for training
    train_dataset = PairwiseComparisonDataset(train_data, num_samples=5000)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 2. Model Initialization
    input_dim = train_data.restaurant_embeddings.shape[1]
    num_train_items = len(train_data.restaurants_df)
    
    # Tabular Model (learns embeddings for specific IDs 0..N-1)
    tabular_model = TabularPL(num_train_items)
    
    # Neural Model (learns from features)
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
        rho_tab_train = evaluate_ranking(tabular_model, train_data, 'tabular')
        rho_neu_train = evaluate_ranking(neural_model, train_data, 'neural')
        
        # Zero-Shot Evaluation (Test Sets)
        rho_2023 = evaluate_ranking(neural_model, test_data_2023, 'neural')
        rho_2025 = evaluate_ranking(neural_model, test_data_2025, 'neural')
        
        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"Loss Tab: {total_loss_tab/len(train_loader):.3f} | "
              f"Loss Neu: {total_loss_neu/len(train_loader):.3f} | "
              f"Train Rho (Tab/Neu): {rho_tab_train:.2f}/{rho_neu_train:.2f} | "
              f"Test 2023: {rho_2023:.2f} | Test 2025: {rho_2025:.2f}")

    print("\n--- Final Verdict ---")
    print("The Tabular model learns the training set well but cannot generalize to new years.")
    print(f"The Neural model generalizes to 2023 ({rho_2023:.2f}) and 2025 ({rho_2025:.2f}) data.")

if __name__ == "__main__":
    main()
