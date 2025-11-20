import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from scipy.stats import spearmanr, pearsonr
from pl import TabularPlackettLuce, NeuralPlackettLuce, EMPlackettLuce, GeneralizedPlackettLuce
import os

def load_data(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def main():
    print("=== Plackett-Luce Online Learning Experiment ===")
    
    # 1. Load Data
    # Use absolute path relative to this script
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pairwise_restaurant_preferences.json")
    if not os.path.exists(data_path):
        print(f"Data file not found at {data_path}. Please run create_pairwise_dataset.py first.")
        return
        
    print(f"Loading data from {data_path}...")
    comparisons = load_data(data_path)
    print(f"Loaded {len(comparisons)} comparisons.")
    
    # 2. Prepare Features (Embeddings)
    print("Computing embeddings for restaurants (features for Neural model)...")
    # We need a map from restaurant ID to its text embedding
    # Extract unique restaurants
    unique_restaurants = {}
    for entry in comparisons:
        r_a = entry['restaurant_a']
        r_b = entry['restaurant_b']
        unique_restaurants[r_a['id']] = r_a['text']
        unique_restaurants[r_b['id']] = r_b['text']
        
    sorted_ids = sorted(unique_restaurants.keys())
    texts = [unique_restaurants[uid] for uid in sorted_ids]
    
    model_st = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model_st.encode(texts)
    
    # Map ID -> Tensor
    id_to_embedding = {uid: torch.tensor(emb) for uid, emb in zip(sorted_ids, embeddings)}
    input_dim = embeddings.shape[1]
    print(f"Computed embeddings for {len(unique_restaurants)} restaurants. Dimension: {input_dim}")
    
    # 3. Initialize Models
    print("\nInitializing Models...")
    # Tabular Model: Learns one scalar per restaurant ID
    tabular_model = TabularPlackettLuce(learning_rate=0.05)
    
    # Neural Model: Learns mapping from Embedding -> Scalar
    neural_model = NeuralPlackettLuce(input_dim=input_dim, hidden_dim=64, learning_rate=0.01)
    
    # EM Model: Batch learner
    em_model = EMPlackettLuce()

    # Generalized Model: Learns Utility + Rationality
    # We have 100 personas (0-99) and 50 items
    gen_model = GeneralizedPlackettLuce(num_items=50, num_voters=100, sgd_all=True)
    
    # 4. Online Learning Loop
    print("\nStarting Online Learning...")
    print("We will process comparisons one by one, predict the winner, then update the model.")
    
    tabular_correct = 0
    neural_correct = 0
    total = 0
    
    tabular_accuracy_history = []
    neural_accuracy_history = []
    
    # Shuffle data to ensure i.i.d assumption holds roughly
    np.random.seed(42)
    np.random.shuffle(comparisons)
    
    for i, comp in enumerate(comparisons):
        # Extract IDs and Features
        id_a = comp['restaurant_a']['id']
        id_b = comp['restaurant_b']['id']
        
        # Extract Persona Info (stored but not used by global models yet)
        persona_id = comp.get('persona_id')
        
        feat_a = id_to_embedding[id_a]
        feat_b = id_to_embedding[id_b]
        
        # Ground Truth
        # In our dataset, winner_id is the ground truth winner
        winner_id = comp['winner_id']
        
        # --- Prediction Step (Before Update) ---
        
        # Tabular Prediction
        prob_a_wins_tab = tabular_model.predict_proba(id_a, id_b)
        pred_winner_tab = id_a if prob_a_wins_tab > 0.5 else id_b
        
        # Neural Prediction
        prob_a_wins_neural = neural_model.predict_proba(feat_a, feat_b)
        pred_winner_neural = id_a if prob_a_wins_neural > 0.5 else id_b
        
        # Check Accuracy
        if pred_winner_tab == winner_id:
            tabular_correct += 1
        
        if pred_winner_neural == winner_id:
            neural_correct += 1
            
        total += 1
        
        # --- Update Step ---
        loser_id = comp['loser_id']
        
        # Tabular Update
        tabular_model.update(winner_id, loser_id)
        
        # Neural Update
        # We need to pass the features of the winner and loser
        feat_winner = id_to_embedding[winner_id]
        feat_loser = id_to_embedding[loser_id]
        neural_model.update(feat_winner, feat_loser)
        
        # EM Accumulation (Batch)
        em_model.add_comparison(winner_id, loser_id, persona_id=persona_id)

        # Generalized Model Accumulation
        gen_model.add_comparison(winner_id, loser_id, persona_id=persona_id)
        
        # Logging
        if total % 100 == 0:
            tab_acc = tabular_correct / total
            neu_acc = neural_correct / total
            tabular_accuracy_history.append(tab_acc)
            neural_accuracy_history.append(neu_acc)
            print(f"Step {total}: Tabular Acc={tab_acc:.3f}, Neural Acc={neu_acc:.3f}")

    print("\n=== Final Results (Online Models) ===")
    print(f"Total Comparisons: {total}")
    print(f"Tabular Model Accuracy: {tabular_correct / total:.4f}")
    print(f"Neural Model Accuracy: {neural_correct / total:.4f}")
    
    # 5. Run EM Model
    print("\n=== Running EM Model (Batch) ===")
    print("Fitting EM model on the full history...")
    em_model.fit(max_iter=20)

    # 6. Run Generalized Model
    print("\n=== Running Generalized Model (Batch Hybrid) ===")
    print("Fitting Generalized model (MM for u, SGD for beta)...")
    # We run a few iterations of the hybrid update
    for _ in range(20):
        gen_model.step()
    
    # Evaluate EM Model on the same dataset (Training Accuracy)
    em_correct = 0
    gen_correct = 0
    oracle_correct = 0
    
    for comp in comparisons:
        id_a = comp['restaurant_a']['id']
        id_b = comp['restaurant_b']['id']
        winner_id = comp['winner_id']
        persona_id = int(comp.get('persona_id', 0))
        
        # EM Prediction
        prob_a = em_model.predict_proba(id_a, id_b)
        pred = id_a if prob_a > 0.5 else id_b
        if pred == winner_id:
            em_correct += 1

        # Generalized Prediction
        prob_a_gen = gen_model.predict_proba(id_a, id_b, persona_id=persona_id)
        pred_gen = id_a if prob_a_gen > 0.5 else id_b
        if pred_gen == winner_id:
            gen_correct += 1
            
        # Oracle Prediction (Ground Truth Utility)
        # The dataset stores the utility of the winner and loser for that specific comparison
        # But strictly, the Oracle checks the true utility values.
        # In our dataset generation, winner_utility and loser_utility ARE the true utilities.
        u_winner = comp['winner_utility']
        u_loser = comp['loser_utility']
        
        # Oracle predicts the one with higher utility should win
        # Since u_winner is the utility of the item that WON in this observation,
        # and u_loser is the utility of the item that LOST,
        # The Oracle predicts the winner correctly if u_winner > u_loser.
        if u_winner > u_loser:
            oracle_correct += 1
            
    print(f"EM Model Accuracy (Training Set): {em_correct / total:.4f}")
    print(f"Generalized Model Accuracy (Training Set): {gen_correct / total:.4f}")
    print(f"Oracle Accuracy (Bayes Optimal): {oracle_correct / total:.4f}")
    
    # --- Ranking Quality Evaluation ---
    print("\n=== Ranking Quality Evaluation (Estimation Error) ===")
    
    # Load Ground Truth
    gt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ground_truth_rankings.json")
    if os.path.exists(gt_path):
        with open(gt_path, 'r') as f:
            ground_truth = json.load(f)
            
        # 1. Global Models vs Aggregated Ranking (Borda)
        print("\n--- Global Models vs Aggregated Ground Truth (Borda) ---")
        borda_ranking = ground_truth['aggregated_rankings']['borda']
        # borda_ranking is a list of item IDs in order of rank (0=best, N-1=worst)
        # We need to convert this to a rank vector: item_id -> rank
        true_ranks_global = np.zeros(len(borda_ranking))
        for rank, item_id in enumerate(borda_ranking):
            true_ranks_global[item_id] = rank + 1
            
        # Helper to get predicted ranks
        def get_predicted_ranks(scores_dict, num_items):
            # scores_dict: item_id -> score
            # Sort items by score descending
            sorted_items = sorted(scores_dict.keys(), key=lambda k: scores_dict[k], reverse=True)
            pred_ranks = np.zeros(num_items)
            for rank, item_id in enumerate(sorted_items):
                if item_id < num_items: # Safety check
                    pred_ranks[item_id] = rank + 1
            return pred_ranks

        # Tabular
        tab_scores = {i: tabular_model.get_score(i) for i in range(50)}
        tab_ranks = get_predicted_ranks(tab_scores, 50)
        rho_tab, _ = spearmanr(true_ranks_global, tab_ranks)
        r_tab, _ = pearsonr(true_ranks_global, tab_ranks)
        print(f"Tabular Model: Spearman Rho = {rho_tab:.4f}, Pearson r (ranks) = {r_tab:.4f}")
        
        # Neural
        # Compute scores for all items
        neural_scores = {}
        for i in range(50):
            feat = id_to_embedding[i].unsqueeze(0)
            with torch.no_grad():
                score = neural_model.forward(feat).item()
            neural_scores[i] = score
        neu_ranks = get_predicted_ranks(neural_scores, 50)
        rho_neu, _ = spearmanr(true_ranks_global, neu_ranks)
        r_neu, _ = pearsonr(true_ranks_global, neu_ranks)
        print(f"Neural Model:  Spearman Rho = {rho_neu:.4f}, Pearson r (ranks) = {r_neu:.4f}")
        
        # EM
        # EM stores gammas. score = log(gamma)
        em_scores = {i: np.log(em_model.gammas.get(i, 1.0)) for i in range(50)}
        em_ranks = get_predicted_ranks(em_scores, 50)
        rho_em, _ = spearmanr(true_ranks_global, em_ranks)
        r_em, _ = pearsonr(true_ranks_global, em_ranks)
        print(f"EM Model:      Spearman Rho = {rho_em:.4f}, Pearson r (ranks) = {r_em:.4f}")
        
        # 2. Generalized Model vs Individual Rankings
        print("\n--- Generalized Model vs Individual Ground Truth ---")
        # For each persona, we predict scores and compare with their specific ground truth
        rhos = []
        rs = []
        
        individual_rankings = ground_truth['individual_rankings']
        
        for p_id in range(100):
            # Get True Ranks for this persona
            # individual_rankings[str(p_id)] is {item_id: rank}
            p_true_ranks_map = individual_rankings[str(p_id)]
            p_true_ranks = np.zeros(50)
            for i in range(50):
                p_true_ranks[i] = p_true_ranks_map.get(str(i), 50) # Default to last if missing
            
            # Get Predicted Scores for this persona
            # Generalized model: score_i = u_i * beta_p (roughly, or just u_i since beta is monotonic scaling)
            # Wait, in Generalized PL, P(i>j) depends on beta * (u_i - u_j).
            # The ranking is determined by u_i. Beta only affects the confidence/probability.
            # So for a standard Generalized PL where beta is a scalar multiplier, the ranking is the SAME for all personas!
            # Unless the model is "Contextual" where u_i depends on persona features.
            # Our GeneralizedPlackettLuce learns ONE set of utilities u_i and ONE beta_p per persona.
            # So it predicts the SAME ranking for everyone, just with different "sharpness".
            # So we should compare the learned utilities 'gamma' against the individual rankings.
            # But wait, if the ground truth has different rankings for different personas (which it does, based on embeddings),
            # then a model that learns a SINGLE utility vector cannot possibly fit everyone perfectly.
            # It will learn the "consensus" ranking.
            # So comparing it to individual rankings will show how well the consensus fits individuals.
            
            # Let's use the model's learned utilities (u)
            # GeneralizedPlackettLuce stores log-utilities in self.u (Tensor)
            gen_scores = {i: gen_model.u[i].item() for i in range(50)}
            gen_ranks = get_predicted_ranks(gen_scores, 50)
            
            rho, _ = spearmanr(p_true_ranks, gen_ranks)
            r, _ = pearsonr(p_true_ranks, gen_ranks)
            rhos.append(rho)
            rs.append(r)
            
        print(f"Generalized Model (Avg across personas): Spearman Rho = {np.mean(rhos):.4f}, Pearson r = {np.mean(rs):.4f}")
        print("Note: Since Generalized PL learns a single utility vector (consensus), this metric measures how well the consensus fits individuals.")
        
    else:
        print("Ground truth rankings file not found. Skipping ranking evaluation.")

    # --- Evaluation on Test Sets (2023, 2025) ---
    print("\n=== Evaluation on Test Sets (Generalization) ===")
    print("Evaluating Neural Model on unseen restaurants from 2023 and 2025...")
    print("Note: Tabular, EM, and Generalized models cannot predict for new items without retraining.")
    
    # Helper to get predicted ranks (redefined here for clarity)
    def get_predicted_ranks_local(scores_dict, num_items):
        sorted_items = sorted(scores_dict.keys(), key=lambda k: scores_dict[k], reverse=True)
        pred_ranks = np.zeros(num_items)
        for rank, item_id in enumerate(sorted_items):
            if item_id < num_items:
                pred_ranks[item_id] = rank + 1
        return pred_ranks

    test_years = ['2023', '2025']
    for year in test_years:
        # Use absolute path
        test_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"test_dataset_{year}.json")
        if not os.path.exists(test_file):
            print(f"Test file {test_file} not found.")
            continue
            
        print(f"\n--- Test Set {year} ---")
        with open(test_file, 'r') as f:
            test_data = json.load(f)
            
        # Ground Truth Ranking (Borda)
        gt_ranking_list = test_data['borda_ranking']
        num_items = len(gt_ranking_list)
        
        # Convert to rank vector: item_id -> rank
        true_ranks = np.zeros(num_items)
        for rank, item_id in enumerate(gt_ranking_list):
            true_ranks[item_id] = rank + 1
            
        # Compute Embeddings for Test Items
        # Reuse model_st from earlier in main()
        test_texts = [item['text'] for item in test_data['restaurants']]
        test_embeddings = model_st.encode(test_texts)
        test_embeddings_tensor = torch.tensor(test_embeddings)
        
        # Predict Scores using Neural Model
        pred_scores = {}
        with torch.no_grad():
            scores = neural_model.forward(test_embeddings_tensor)
            
        for i in range(num_items):
            pred_scores[i] = scores[i].item()
            
        # Compute Predicted Ranks
        pred_ranks = get_predicted_ranks_local(pred_scores, num_items)
        
        # Metrics
        rho, _ = spearmanr(true_ranks, pred_ranks)
        r, _ = pearsonr(true_ranks, pred_ranks)
        
        print(f"Neural Model on {year} Data (Zero-Shot):")
        print(f"Spearman Rho = {rho:.4f}")
        print(f"Pearson r (ranks) = {r:.4f}")

    # Scientific Commentary
    print("\n=== Scientific Commentary ===")
    print("1. **Oracle Accuracy**: This represents the theoretical upper bound (Bayes Error Rate).")
    print("   Since the data is probabilistic (generated via Bradley-Terry), the 'better' item doesn't always win.")
    print("   The Oracle accuracy reflects the noise level in the dataset.")
    print("2. **Generalized Model**: This model learns a separate 'rationality' (beta) parameter for each persona.")
    print("   If some personas are more consistent than others, this model can weigh their votes accordingly.")
    print("   In this synthetic dataset, all comparisons were generated with the same beta=10, so we expect")
    print("   the Generalized model to perform similarly to the standard EM model, potentially slightly worse due to overfitting.")
    print("3. **Convergence**: Observe how the accuracy stabilizes over time.")
    print("4. **Cold Start**: The Neural model has an advantage for new items (cold start) because it uses content features.")
    print("   The Tabular model starts with score 0 for everyone and must see an item to learn about it.")
    print("   However, in this experiment, we have a fixed set of 50 restaurants, so the Tabular model catches up quickly.")
    print("5. **Capacity**: The Tabular model has N parameters (one per item). The Neural model has fixed parameters independent of N.")
    print("   For very large N, Neural is more memory efficient.")
    print("6. **Generalization**: If we added a new restaurant #501, the Neural model could immediately predict its score based on description.")
    print("   The Tabular model would have no clue.")
    print("7. **EM vs SGD**: The EM algorithm finds the Maximum Likelihood Estimate (MLE) for the static dataset.")
    print("   The Tabular SGD model should converge to the same solution as EM if the learning rate decays appropriately.")
    print("   EM is more stable but requires batch processing. SGD is online and adaptive.")

if __name__ == "__main__":
    main()
