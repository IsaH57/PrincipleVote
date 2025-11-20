import numpy as np
import torch
from datasets import load_from_disk
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json
import random
import os
import pandas as pd
from pref_voting.profiles import Profile
from pref_voting.scoring_methods import borda_ranking

def create_text_representation_persona(example):
    parts = [
        example.get('persona', ''),
        example.get('culinary_persona', ''),
        example.get('cultural_background', '')
    ]
    return " ".join([p for p in parts if p])

def create_text_representation_restaurant_csv(row):
    name = row.get('Restaurant Name', 'Unknown Restaurant')
    desc = row.get('Description', '')
    return f"{name}: {desc}"

def main():
    print("Loading personas...")
    try:
        personas_ds = load_from_disk("/home/ra63hik/dppo/data/nemotron_personas")['train']
    except Exception as e:
        print(f"Error loading personas: {e}")
        return

    # 1. Sample SAME 100 personas
    np.random.seed(42)
    random.seed(42)
    
    persona_indices = np.random.choice(len(personas_ds), 100, replace=False)
    sampled_personas = personas_ds.select(persona_indices)
    
    print("Embedding personas...")
    persona_texts = [create_text_representation_persona(p) for p in sampled_personas]
    model = SentenceTransformer('all-MiniLM-L6-v2')
    persona_embeddings = model.encode(persona_texts)
    
    years = ['2023', '2025']
    files = {'2023': 'nyt_restaurant_2023.csv', '2025': 'nyt_restaurants_2025.csv'}
    
    for year in years:
        filename = files[year]
        if not os.path.exists(filename):
            print(f"File {filename} not found. Skipping.")
            continue
            
        print(f"\nProcessing {year} data from {filename}...")
        df = pd.read_csv(filename)
        print(f"Loaded {len(df)} restaurants.")
        
        # Prepare text
        restaurant_texts = df.apply(create_text_representation_restaurant_csv, axis=1).tolist()
        restaurant_names = df['Restaurant Name'].tolist()
        
        # Embed
        print("Embedding restaurants...")
        restaurant_embeddings = model.encode(restaurant_texts)
        
        # Compute Utilities
        print("Computing utilities...")
        utilities = cosine_similarity(persona_embeddings, restaurant_embeddings)
        num_personas, num_items = utilities.shape
        
        # Compute Aggregated Ranking (Borda)
        print("Computing Borda ranking...")
        rankings_list = []
        for p_idx in range(num_personas):
            u = utilities[p_idx]
            ranking = tuple(np.argsort(u)[::-1])
            rankings_list.append(ranking)
            
        counts_list = [1] * num_personas
        prof = Profile(rankings_list, counts_list)
        
        br = borda_ranking(prof)
        
        # Convert to list of IDs
        sorted_items = sorted(br.rmap.keys(), key=lambda k: br.rmap[k])
        borda_ranking_list = [int(i) for i in sorted_items]
        
        # Save Data
        output_data = {
            "year": year,
            "restaurants": [
                {"id": i, "name": name, "text": text} 
                for i, (name, text) in enumerate(zip(restaurant_names, restaurant_texts))
            ],
            "utilities": utilities.tolist(),
            "borda_ranking": borda_ranking_list
        }
        
        output_path = f"test_dataset_{year}.json"
        print(f"Saving to {output_path}...")
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
            
    print("\nDone creating test datasets.")

if __name__ == "__main__":
    main()
