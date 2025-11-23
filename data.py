"""
Data loading and synthetic preference generation.
"""

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class RestaurantData:
    """
    Manages the raw restaurant and persona data.
    Computes embeddings using a pre-trained Transformer.
    """
    def __init__(self, data_dir="data", restaurants_file="restaurants.csv", model_name='all-MiniLM-L6-v2'):
        self.data_dir = data_dir
        self.model = SentenceTransformer(model_name)
        
        # Load CSVs
        self.personas_df = pd.read_csv(os.path.join(data_dir, "personas.csv"))
        self.restaurants_df = pd.read_csv(os.path.join(data_dir, restaurants_file))
        
        # Pre-compute text representations
        self.restaurants_df['text'] = self.restaurants_df.apply(self._restaurant_text, axis=1)
        
        print(f"Loaded {len(self.personas_df)} personas and {len(self.restaurants_df)} restaurants from {restaurants_file}.")
        print("Computing embeddings... (this may take a moment)")
        
        # Compute Embeddings
        # We only use a subset of personas to keep things fast for this demo
        self.personas_df = self.personas_df.sample(n=min(100, len(self.personas_df)), random_state=42).reset_index(drop=True)
        self.personas_df['text'] = self.personas_df.apply(self._persona_text, axis=1)
        
        self.persona_embeddings = self.model.encode(self.personas_df['text'].tolist())
        self.restaurant_embeddings = self.model.encode(self.restaurants_df['text'].tolist())
        
        # Compute Ground Truth Utilities (Cosine Similarity)
        # Shape: (num_personas, num_restaurants)
        self.utilities = cosine_similarity(self.persona_embeddings, self.restaurant_embeddings)
        
    def _persona_text(self, row):
        parts = [
            str(row.get('persona', '')),
            str(row.get('culinary_persona', '')),
            str(row.get('cultural_background', ''))
        ]
        return " ".join([p for p in parts if p and p != 'nan'])

    def _restaurant_text(self, row):
        # Handle different column names across datasets
        name = row.get('restaurant_name') or row.get('Restaurant Name') or 'Unknown'
        cuisine = row.get('cuisine_type') or row.get('Cuisine') or 'Unknown Cuisine'
        desc = row.get('description') or row.get('Description') or ''
        return f"{name}: {desc}"

class PairwiseComparisonDataset(Dataset):
    """
    PyTorch Dataset for pairwise comparisons.
    Generates synthetic comparisons based on the Bradley-Terry model
    using the ground truth utilities from RestaurantData.
    """
    def __init__(self, restaurant_data, num_samples=10000, beta=10.0, seed=42):
        self.r_data = restaurant_data
        self.num_samples = num_samples
        self.beta = beta
        self.rng = np.random.RandomState(seed)
        
        self.comparisons = self._generate_data()
        
    def _generate_data(self):
        data = []
        num_personas = len(self.r_data.personas_df)
        num_restaurants = len(self.r_data.restaurants_df)
        
        print(f"Generating {self.num_samples} pairwise comparisons...")
        
        for _ in range(self.num_samples):
            # Pick a random persona
            p_idx = self.rng.randint(0, num_personas)
            
            # Pick two distinct restaurants
            r1_idx, r2_idx = self.rng.choice(num_restaurants, 2, replace=False)
            
            u1 = self.r_data.utilities[p_idx, r1_idx]
            u2 = self.r_data.utilities[p_idx, r2_idx]
            
            # Bradley-Terry Probability
            # P(1 > 2) = sigmoid(beta * (u1 - u2))
            prob_1_wins = 1.0 / (1.0 + np.exp(-self.beta * (u1 - u2)))
            
            if self.rng.rand() < prob_1_wins:
                winner, loser = r1_idx, r2_idx
            else:
                winner, loser = r2_idx, r1_idx
                
            data.append({
                'winner_id': winner,
                'loser_id': loser,
                'winner_feat': self.r_data.restaurant_embeddings[winner],
                'loser_feat': self.r_data.restaurant_embeddings[loser]
            })
            
        return data

    def __len__(self):
        return len(self.comparisons)

    def __getitem__(self, idx):
        item = self.comparisons[idx]
        return {
            'winner_id': torch.tensor(item['winner_id'], dtype=torch.long),
            'loser_id': torch.tensor(item['loser_id'], dtype=torch.long),
            'winner_feat': torch.tensor(item['winner_feat'], dtype=torch.float32),
            'loser_feat': torch.tensor(item['loser_feat'], dtype=torch.float32)
        }
