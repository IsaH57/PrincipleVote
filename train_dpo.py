"""
RLHF (DPO) Training Script.
Fine-tunes a small LLM (GPT-2) to rank restaurants based on pairwise comparisons.
"""

import torch
import torch.nn.functional as F
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
import numpy as np
from scipy.stats import spearmanr
import copy
from tqdm import tqdm

from data import RestaurantData, PairwiseComparisonDataset

MAX_LEN = 20 # Max length for generation

def get_log_probs(model, tokenizer, prompts, responses, device):
    """
    Computes the log probabilities of the responses given the prompts.
    Returns log P(prompt + response). 
    Since prompts are identical for chosen/rejected, the prompt probability cancels out in DPO difference.
    """
    # Concatenate prompt + response
    inputs = [p + r for p, r in zip(prompts, responses)]
    
    # Tokenize
    encodings = tokenizer(inputs, return_tensors='pt', padding=True, truncation=True, max_length=MAX_LEN).to(device)
    input_ids = encodings.input_ids
    attention_mask = encodings.attention_mask
    
    # Forward pass
    outputs = model(input_ids, attention_mask=attention_mask)
    logits = outputs.logits # (B, Seq, Vocab)
    
    # Shift logits and labels to align
    # logits[:, :-1] predicts input_ids[:, 1:]
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    
    # Compute log probs for each token
    token_log_probs = F.log_softmax(shift_logits, dim=-1)
    
    # Gather log probs of the actual tokens
    gathered_log_probs = torch.gather(token_log_probs, -1, shift_labels.unsqueeze(-1)).squeeze(-1)
    
    # Mask padding
    mask = attention_mask[:, 1:] # Shift mask
    seq_log_probs = (gathered_log_probs * mask).sum(dim=1)
    
    return seq_log_probs

def evaluate_dpo_ranking(model, tokenizer, r_data, device):
    """
    Evaluates the model by calculating P(Name | "The best restaurant is") for all restaurants.
    """
    model.eval()
    prompt = "The best restaurant is"
    
    names = []
    # Handle different column names
    name_col = None
    for col in ['restaurant_name']:
        if col in r_data.restaurants_df.columns:
            name_col = col
            break
    
    if not name_col:
        return 0.0
        
    names = r_data.restaurants_df[name_col].tolist()
    prompts = [prompt] * len(names)
    # Add leading space to names for GPT-2
    responses = [" " + n for n in names]
    
    with torch.no_grad():
        log_probs = get_log_probs(model, tokenizer, prompts, responses, device)
        scores = log_probs.cpu().numpy()
        
    # Ground Truth (Mean Utility)
    mean_utilities = r_data.utilities.mean(axis=0)
    
    # Correlation
    rho, _ = spearmanr(scores, mean_utilities)
    return rho

def main():
    # Config
    MODEL_NAME = "gpt2"
    BATCH_SIZE = 16
    LR = 1e-4 # Low LR for fine-tuning
    BETA = 0.1
    EPOCHS = 2000
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"--- DPO Training ({DEVICE}) ---")
    
    # 1. Load Data
    print("Loading Data...")
    train_data = RestaurantData(restaurants_file="restaurants.csv")
    test_data_2023 = RestaurantData(restaurants_file="restaurants_2023.csv")
    test_data_2025 = RestaurantData(restaurants_file="restaurants_2025.csv")
    
    # Generate Comparisons
    # We need names for DPO
    train_dataset = PairwiseComparisonDataset(train_data, num_samples=2000)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Helper to get name from ID
    name_col = 'restaurant_name' if 'restaurant_name' in train_data.restaurants_df else 'Restaurant Name'
    id_to_name = train_data.restaurants_df[name_col].to_dict()
    
    # DEBUG: Print samples
    DEBUG = True
    if DEBUG:
        print("\n--- DEBUG: Data Samples ---")
        print("Training Samples (Pairwise):")
        for i in range(min(3, len(train_dataset))):
            sample = train_dataset[i]
            w_name = id_to_name[sample['winner_id'].item()]
            l_name = id_to_name[sample['loser_id'].item()]
            print(f"  {i+1}. Winner: {w_name} | Loser: {l_name}")
            
        print("\nTest 2023 Samples (Names):")
        t_name_col = 'restaurant_name' if 'restaurant_name' in test_data_2023.restaurants_df else 'Restaurant Name'
        test_names_2023 = test_data_2023.restaurants_df[t_name_col].head(3).tolist()
        for n in test_names_2023:
            print(f"  - {n}")
        print("---------------------------\n")
    
    # 2. Load Models
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    policy_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
    ref_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
    
    # CRITICAL: Freeze reference model parameters
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False
    
    optimizer = optim.AdamW(policy_model.parameters(), lr=LR)
    
    # 3. Training Loop
    print("Starting DPO Fine-tuning...")
    prompt_text = "The best restaurant is"
    
    accumulation_steps = 4 # Simulate larger batch size
    total_steps = (len(train_loader) // accumulation_steps) * EPOCHS
    warmup_steps = min(100, total_steps // 10)
    
    # Warmup + Cosine Annealing Scheduler
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        return 0.5 * (1 + np.cos(np.pi * (step - warmup_steps) / (total_steps - warmup_steps)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    for epoch in range(EPOCHS):
        policy_model.train()
        total_loss = 0
        optimizer.zero_grad()
        
        for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            w_ids = batch['winner_id'].tolist()
            l_ids = batch['loser_id'].tolist()
            
            prompts = [prompt_text] * len(w_ids)
            chosen_texts = [" " + id_to_name[i] for i in w_ids]
            rejected_texts = [" " + id_to_name[i] for i in l_ids]
            
            # Compute Log Probs
            # Policy
            policy_chosen_logps = get_log_probs(policy_model, tokenizer, prompts, chosen_texts, DEVICE)
            policy_rejected_logps = get_log_probs(policy_model, tokenizer, prompts, rejected_texts, DEVICE)
            
            # Reference (No Grad)
            with torch.no_grad():
                ref_chosen_logps = get_log_probs(ref_model, tokenizer, prompts, chosen_texts, DEVICE)
                ref_rejected_logps = get_log_probs(ref_model, tokenizer, prompts, rejected_texts, DEVICE)
            
            # DPO Loss
            # log(sigmoid(beta * (log(pi_w/ref_w) - log(pi_l/ref_l))))
            logits = BETA * ( (policy_chosen_logps - ref_chosen_logps) - (policy_rejected_logps - ref_rejected_logps) )
            loss = -F.logsigmoid(logits).mean()
            
            loss = loss / accumulation_steps
            loss.backward()
            
            if (i + 1) % accumulation_steps == 0:
                # Debug: Check gradients
                total_norm = 0
                for p in policy_model.parameters():
                    if p.grad is not None:
                        total_norm += p.grad.data.norm(2).item() ** 2
                total_norm = total_norm ** 0.5
                
                torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                # Print grad norm occasionally for debugging
                if i % (accumulation_steps * 10) == 0:
                    current_lr = optimizer.param_groups[0]['lr']
                    print(f"  Step {i//accumulation_steps} | Grad Norm: {total_norm:.4f} | LR: {current_lr:.6f}")
            
            total_loss += loss.item() * accumulation_steps
            
        # Evaluation
        rho_train = evaluate_dpo_ranking(policy_model, tokenizer, train_data, DEVICE)
        rho_2023 = evaluate_dpo_ranking(policy_model, tokenizer, test_data_2023, DEVICE)
        rho_2025 = evaluate_dpo_ranking(policy_model, tokenizer, test_data_2025, DEVICE)
        
        print(f"Epoch {epoch+1} | Loss: {total_loss/len(train_loader):.4f}")
        print(f"Spearman Rho - Train: {rho_train:.3f} | Test 2023: {rho_2023:.3f} | Test 2025: {rho_2025:.3f}")

if __name__ == "__main__":
    main()
