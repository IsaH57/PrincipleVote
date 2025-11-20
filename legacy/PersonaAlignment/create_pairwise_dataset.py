import numpy as np
import torch
from datasets import load_from_disk, load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json
import random
import os
from pref_voting.profiles import Profile
from pref_voting.scoring_methods import borda_ranking, plurality_ranking
from pref_voting.c1_methods import copeland_ranking

def create_text_representation_persona(example):
    # Combine relevant fields for a rich representation
    parts = [
        example.get('persona', ''),
        example.get('culinary_persona', ''),
        example.get('cultural_background', '')
    ]
    return " ".join([p for p in parts if p])

def create_text_representation_restaurant(example):
    name = example.get('restaurant_name', 'Unknown Restaurant')
    cuisine = example.get('cuisine_type', 'Unknown Cuisine')
    desc = example.get('description', '')
    return f"{name} ({cuisine}): {desc}"

def main():
    print("Loading datasets...")
    try:
        # Try loading from local CSVs first
        personas_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nemotron_personas.csv")
        restaurants_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nyt_restaurants_2024.csv")
        
        print("Loading from local CSV files...")
        personas_ds = load_dataset('csv', data_files=personas_csv)['train']
        restaurants_ds = load_dataset('csv', data_files=restaurants_csv)['train']
    except Exception as e:
        print(f"Error loading datasets: {e}. Load datasets from HuggingFace: 'nemotron_personas' and 'nytimes_best_restaurants_2024'")
        return

    print(f"Total Personas: {len(personas_ds)}")
    print(f"Total Restaurants: {len(restaurants_ds)}")

    # 1. Sample 100 personas
    np.random.seed(42)
    random.seed(42)
    
    persona_indices = np.random.choice(len(personas_ds), 100, replace=False)
    sampled_personas = personas_ds.select(persona_indices)
    
    # 2. Prepare text for embedding
    print("Preparing text for embedding...")
    persona_texts = [create_text_representation_persona(p) for p in sampled_personas]
    restaurant_texts = [create_text_representation_restaurant(r) for r in restaurants_ds]
    
    # 3. Compute Embeddings
    print("Loading SentenceTransformer...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("Embedding personas...")
    persona_embeddings = model.encode(persona_texts)
    
    print("Embedding restaurants...")
    restaurant_embeddings = model.encode(restaurant_texts)
    
    # 4. Compute Utilities (Cosine Similarity)
    print("Computing utilities...")
    # Shape: (100, 500)
    utilities = cosine_similarity(persona_embeddings, restaurant_embeddings)
    
    # 5. Generate Pairwise Comparisons
    print("Generating pairwise comparisons...")
    num_comparisons = 50000 # Generate a good amount of data
    dataset = []
    
    # Sampling strategy: 'deterministic' or 'probabilistic'
    sampling_strategy = 'probabilistic' 
    print(f"Using {sampling_strategy} sampling strategy.")
    
    # We can just sample randomly as requested
    for _ in range(num_comparisons):
        # Pick a random persona index (0 to 99)
        p_idx = random.randint(0, 99)
        persona_data = sampled_personas[p_idx]
        
        # Pick two distinct restaurants
        r1_idx, r2_idx = random.sample(range(len(restaurants_ds)), 2)
        
        u1 = utilities[p_idx, r1_idx]
        u2 = utilities[p_idx, r2_idx]
        
        # Determine winner
        if sampling_strategy == 'deterministic':
            if u1 > u2:
                winner_idx, loser_idx = r1_idx, r2_idx
                winner_score, loser_score = u1, u2
            else:
                winner_idx, loser_idx = r2_idx, r1_idx
                winner_score, loser_score = u2, u1
        else:
            # Probabilistic (Bradley-Terry)
            # P(1 > 2) = 1 / (1 + exp(-(u1 - u2)))
            # Note: Utilities are cosine similarities [-1, 1]. 
            # We might want to scale them to make the sigmoid more sensitive, 
            # e.g., beta * (u1 - u2). Let's use beta=10 for reasonable spread.
            beta = 10.0
            prob_1_wins = 1.0 / (1.0 + np.exp(-beta * (u1 - u2)))
            
            if random.random() < prob_1_wins:
                winner_idx, loser_idx = r1_idx, r2_idx
                winner_score, loser_score = u1, u2
            else:
                winner_idx, loser_idx = r2_idx, r1_idx
                winner_score, loser_score = u2, u1
            
        # Store data
        # We need to store enough info to train the model later.
        # The Neural Plackett-Luce needs features. 
        # We can store the text, or pre-computed embeddings?
        # The prompt says "attributes of an object are given and we compute a neural representation".
        # So we should probably store the raw text or attributes so the model can process them, 
        # OR store the embeddings we just computed if we want to use those as features.
        # Let's store the text attributes so it's more general.
        
        entry = {
            "persona_id": str(p_idx), # ID within our sample
            "persona_text": persona_texts[p_idx],
            "restaurant_a": {
                "id": r1_idx,
                "text": restaurant_texts[r1_idx],
                "name": restaurants_ds[r1_idx]['restaurant_name'],
                "cuisine": restaurants_ds[r1_idx]['cuisine_type'],
                "description": restaurants_ds[r1_idx]['description']
            },
            "restaurant_b": {
                "id": r2_idx,
                "text": restaurant_texts[r2_idx],
                "name": restaurants_ds[r2_idx]['restaurant_name'],
                "cuisine": restaurants_ds[r2_idx]['cuisine_type'],
                "description": restaurants_ds[r2_idx]['description']
            },
            "winner_id": winner_idx, # Global restaurant ID
            "loser_id": loser_idx,
            "winner_utility": float(winner_score),
            "loser_utility": float(loser_score)
        }
        dataset.append(entry)
        
    # Save dataset
    # Use absolute path relative to this script to avoid CWD issues
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pairwise_restaurant_preferences.json")
    print(f"Saving {len(dataset)} comparisons to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=2)
        
    # --- Compute and Save Ground Truth Rankings ---
    print("Computing ground truth rankings...")
    
    # utilities shape: (100, 50)
    num_personas, num_items = utilities.shape
    
    # 1. Individual Rankings
    # For each persona, we want the rank of each item.
    # argsort gives indices that sort the array.
    # To get ranks (0 = worst, N-1 = best), we can use argsort twice.
    # Or better, let's store the actual rank position (1 = best, N = worst).
    # If utilities are high = good, then we sort descending.
    
    individual_rankings = {} # persona_id -> {item_id -> rank}
    
    for p_idx in range(num_personas):
        # Get utilities for this persona
        u = utilities[p_idx]
        # Sort indices descending by utility
        sorted_indices = np.argsort(u)[::-1]
        
        # Create rank map: item_id -> rank (1-based)
        ranks = {}
        for rank, item_idx in enumerate(sorted_indices):
            ranks[int(item_idx)] = rank + 1
        
        individual_rankings[str(p_idx)] = ranks

    # 2. Aggregated Rankings using pref_voting
    print("Computing aggregated rankings using pref_voting...")
    
    # Create Profile
    rankings_list = []
    for p_idx in range(num_personas):
        u = utilities[p_idx]
        # Sort indices descending by utility to get ranking (best first)
        ranking = tuple(np.argsort(u)[::-1])
        rankings_list.append(ranking)
        
    counts_list = [1] * num_personas
    prof = Profile(rankings_list, counts_list)
    
    # Helper to convert Ranking object to list of item IDs (best to worst)
    # Note: pref_voting Ranking objects handle ties. 
    # If we want a strict ordering for the list, we can sort by rank.
    # Ties will be broken arbitrarily or by ID if we just sort by rank.
    def ranking_to_list(r_obj):
        # r_obj.rmap maps item_id -> rank (1 is best)
        # Sort items by rank
        sorted_items = sorted(r_obj.rmap.keys(), key=lambda k: r_obj.rmap[k])
        return [int(i) for i in sorted_items]

    # Borda
    br = borda_ranking(prof)
    borda_ranking_list = ranking_to_list(br)
    
    # Plurality
    pr = plurality_ranking(prof)
    plurality_ranking_list = ranking_to_list(pr)
    
    # Copeland
    cr = copeland_ranking(prof)
    copeland_ranking_list = ranking_to_list(cr)
    
    ground_truth = {
        "individual_rankings": individual_rankings,
        "aggregated_rankings": {
            "borda": borda_ranking_list,
            "plurality": plurality_ranking_list,
            "copeland": copeland_ranking_list
        },
        "utilities": utilities.tolist() # Save raw utilities too for Pearson correlation
    }
    
    gt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ground_truth_rankings.json")
    print(f"Saving ground truth rankings to {gt_path}...")
    with open(gt_path, 'w') as f:
        json.dump(ground_truth, f, indent=2)
    
    print("Done!")

if __name__ == "__main__":
    main()
