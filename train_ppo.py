"""
PPO Training Script.
Implements RLHF (Reward Modeling + PPO) from scratch for the restaurant ranking task.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
import numpy as np
from tqdm import tqdm
import copy
from scipy.stats import spearmanr
from torch.distributions import Categorical

from data import RestaurantData, PairwiseComparisonDataset

# --- Configuration ---
MODEL_NAME = "gpt2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN = 20 # Max length for generation

def get_reward_score(model, tokenizer, texts, device):
    """Computes scalar score for a list of texts."""
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=MAX_LEN).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.logits.squeeze(-1) # (B,)

def evaluate_ranking(model, tokenizer, r_data, device):
    """Evaluates Policy by scoring all restaurant names."""
    model.eval()
    prompt = "The best restaurant is"
    
    # Get all names
    name_col = None
    for col in ['restaurant_name']:
        if col in r_data.restaurants_df.columns:
            name_col = col
            break
    names = r_data.restaurants_df[name_col].tolist()
    
    prompts = [prompt] * len(names)
    responses = [" " + n for n in names]
    
    # Compute log probs of the names
    # We reuse the logic from DPO script for evaluation consistency
    inputs = [p + r for p, r in zip(prompts, responses)]
    encodings = tokenizer(inputs, return_tensors='pt', padding=True, truncation=True, max_length=MAX_LEN).to(device)
    
    with torch.no_grad():
        outputs = model(encodings.input_ids, attention_mask=encodings.attention_mask)
        logits = outputs.logits[:, :-1, :]
        labels = encodings.input_ids[:, 1:]
        
        token_log_probs = F.log_softmax(logits, dim=-1)
        gathered = torch.gather(token_log_probs, -1, labels.unsqueeze(-1)).squeeze(-1)
        
        # Mask padding
        mask = encodings.attention_mask[:, 1:]
        seq_log_probs = (gathered * mask).sum(dim=1)
        
        # Subtract prompt log prob (approximate/simplified for speed)
        # Actually, just using the full sequence log prob is a good enough proxy for ranking 
        # if the prompt is identical.
        scores = seq_log_probs.cpu().numpy()

    mean_utilities = r_data.utilities.mean(axis=0)
    rho, _ = spearmanr(scores, mean_utilities)
    return rho

# --- 1. Reward Model Training ---

def train_reward_model(rm, tokenizer, train_loader, device, epochs=1):
    print("Training Reward Model...")
    optimizer = optim.AdamW(rm.parameters(), lr=1e-5)
    rm.train()
    
    prompt = "The best restaurant is"
    
    for epoch in range(epochs):
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"RM Epoch {epoch+1}"):
            w_ids = batch['winner_id'].tolist()
            l_ids = batch['loser_id'].tolist()
            
            # We need to map IDs back to text. 
            # The dataset provides features, but we need text for GPT2.
            # We'll access the dataframe directly via the dataset object hack or pass it in.
            # For now, let's assume we can get text from the batch if we modify dataset?
            # No, let's just use the global id_to_name map.
            pass 
            
            # Note: This function assumes id_to_name is available in scope or passed.
            # We will handle this in main.
            
    # Since we need id_to_name inside, let's just put the loop in main or pass it.
    pass

# --- 2. PPO Implementation ---

def ppo_step(policy, value_model, ref_model, reward_model, tokenizer, optimizer, batch_size=16, beta=0.05, ppo_epochs=4):
    # 1. Rollout
    prompt = "The best restaurant is"
    prompts = [prompt] * batch_size
    
    policy.eval()
    inputs = tokenizer(prompts, return_tensors='pt', padding=True).to(DEVICE)
    
    # Generate
    with torch.no_grad():
        gen_outputs = policy.generate(
            **inputs, 
            max_new_tokens=10, 
            do_sample=True, 
            top_k=50, 
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode
    full_texts = tokenizer.batch_decode(gen_outputs, skip_special_tokens=True)
    
    # 2. Compute Rewards
    rm_scores = get_reward_score(reward_model, tokenizer, full_texts, DEVICE)
    
    # Prepare inputs
    attention_mask = (gen_outputs != tokenizer.eos_token_id).long()
    
    # Compute Old Log Probs (for Ratio)
    with torch.no_grad():
        ref_outputs = ref_model(gen_outputs, attention_mask=attention_mask)
        ref_logits = ref_outputs.logits
        ref_logprobs = F.log_softmax(ref_logits[:, :-1, :], dim=-1)
        gen_tokens = gen_outputs[:, 1:]
        ref_token_logprobs = torch.gather(ref_logprobs, -1, gen_tokens.unsqueeze(-1)).squeeze(-1)
        
        policy_outputs = policy(gen_outputs, attention_mask=attention_mask)
        policy_logits = policy_outputs.logits
        policy_logprobs = F.log_softmax(policy_logits[:, :-1, :], dim=-1)
        old_policy_token_logprobs = torch.gather(policy_logprobs, -1, gen_tokens.unsqueeze(-1)).squeeze(-1)

    # KL Penalty
    kl_penalty = old_policy_token_logprobs - ref_token_logprobs
    total_kl = kl_penalty.sum(dim=1)
    
    # Clamp KL to avoid massive negative rewards
    total_kl = torch.clamp(total_kl, min=0, max=10.0)

    # Total Reward
    # Normalize RM scores roughly to keep them in a reasonable range if they aren't already
    # But for now, just clamping KL is a good safety.
    total_rewards = rm_scores - beta * total_kl
    
    # Debug stats (optional, but helpful)
    if np.random.rand() < 0.05:
        print(f"  [Debug] RM: {rm_scores.mean().item():.2f} | KL: {total_kl.mean().item():.2f} | Rew: {total_rewards.mean().item():.2f}")

    # 3. PPO Update Loop
    last_loss = 0
    
    for _ in range(ppo_epochs):
        # Forward Policy
        policy.train()
        policy_outputs = policy(gen_outputs, attention_mask=attention_mask)
        policy_logits = policy_outputs.logits
        policy_logprobs = F.log_softmax(policy_logits[:, :-1, :], dim=-1)
        new_policy_token_logprobs = torch.gather(policy_logprobs, -1, gen_tokens.unsqueeze(-1)).squeeze(-1)
        
        # Forward Value
        value_model.train()
        values = value_model(gen_outputs, attention_mask=attention_mask).logits.squeeze(-1)
        
        # Calculate Advantage
        # Detach targets!
        targets = total_rewards.detach()
        advantages = targets - values.detach()
        
        # Normalize Advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Ratio
        ratio = torch.exp(new_policy_token_logprobs - old_policy_token_logprobs)
        
        # Broadcast Advantage
        adv_expanded = advantages.unsqueeze(-1).expand_as(new_policy_token_logprobs)
        
        # Surrogate Loss
        surr1 = ratio * adv_expanded
        surr2 = torch.clamp(ratio, 1.0 - 0.2, 1.0 + 0.2) * adv_expanded
        
        mask = attention_mask[:, 1:]
        policy_loss = -torch.min(surr1, surr2)
        policy_loss = (policy_loss * mask).sum() / mask.sum()
        
        # Value Loss - Use Huber Loss (SmoothL1) for stability
        value_loss = F.smooth_l1_loss(values, targets)
        
        # Entropy Bonus
        probs = torch.softmax(policy_logits[:, :-1, :], dim=-1)
        log_probs = F.log_softmax(policy_logits[:, :-1, :], dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)
        entropy_loss = -0.01 * (entropy * mask).sum() / mask.sum()
        
        loss = policy_loss + 0.1 * value_loss + entropy_loss
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(value_model.parameters(), 1.0)
        optimizer.step()
        
        last_loss = loss.item()
        
    return last_loss


def main():
    print(f"--- PPO Training ({DEVICE}) ---")
    
    # 1. Load Data
    train_data = RestaurantData(restaurants_file="restaurants.csv")
    test_data_2023 = RestaurantData(restaurants_file="restaurants_2023.csv")
    test_data_2025 = RestaurantData(restaurants_file="restaurants_2025.csv")
    
    # Map ID to Name
    name_col = 'restaurant_name' if 'restaurant_name' in train_data.restaurants_df else 'Restaurant Name'
    id_to_name = train_data.restaurants_df[name_col].to_dict()
    
    # Dataset for RM
    train_dataset = PairwiseComparisonDataset(train_data, num_samples=2000)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    
    # 2. Initialize Models
    print("Initializing Models...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Reward Model (Scalar output)
    reward_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=1).to(DEVICE)
    reward_model.config.pad_token_id = tokenizer.eos_token_id
    
    # Policy & Ref
    policy_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
    ref_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
    ref_model.eval()
    
    # Value Model (Scalar output)
    value_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=1).to(DEVICE)
    value_model.config.pad_token_id = tokenizer.eos_token_id
    
    # 3. Train Reward Model
    print("\nStep 1: Training Reward Model...")
    rm_optimizer = optim.AdamW(reward_model.parameters(), lr=1e-5)
    prompt = "The best restaurant is"
    
    for epoch in range(10):
        reward_model.train()
        total_loss = 0
        for batch in tqdm(train_loader):
            w_ids = batch['winner_id'].tolist()
            l_ids = batch['loser_id'].tolist()
            
            w_texts = [prompt + " " + id_to_name[i] for i in w_ids]
            l_texts = [prompt + " " + id_to_name[i] for i in l_ids]
            
            # Forward
            w_inputs = tokenizer(w_texts, return_tensors='pt', padding=True, truncation=True).to(DEVICE)
            l_inputs = tokenizer(l_texts, return_tensors='pt', padding=True, truncation=True).to(DEVICE)
            
            w_scores = reward_model(**w_inputs).logits
            l_scores = reward_model(**l_inputs).logits
            
            # Loss: -log(sigmoid(w - l))
            loss = -F.logsigmoid(w_scores - l_scores).mean()
            
            rm_optimizer.zero_grad()
            loss.backward()
            rm_optimizer.step()
            total_loss += loss.item()
            
        print(f"RM Loss: {total_loss/len(train_loader):.4f}")
        
    # 4. Train PPO
    print("\nStep 2: Training Policy with PPO...")
    ppo_optimizer = optim.AdamW(list(policy_model.parameters()) + list(value_model.parameters()), lr=1e-5)
    
    for epoch in range(10):
        total_loss = 0
        num_steps = 50 # Number of PPO steps
        
        for _ in tqdm(range(num_steps), desc=f"PPO Epoch {epoch+1}"):
            loss = ppo_step(policy_model, value_model, ref_model, reward_model, tokenizer, ppo_optimizer, batch_size=16)
            
            # ppo_optimizer.step() # Handled inside ppo_step now
            total_loss += loss
            
        # Evaluation
        rho_train = evaluate_ranking(policy_model, tokenizer, train_data, DEVICE)
        rho_2023 = evaluate_ranking(policy_model, tokenizer, test_data_2023, DEVICE)
        rho_2025 = evaluate_ranking(policy_model, tokenizer, test_data_2025, DEVICE)
        
        print(f"Epoch {epoch+1} | Loss: {total_loss/num_steps:.4f}")
        print(f"Spearman Rho - Train: {rho_train:.3f} | Test 2023: {rho_2023:.3f} | Test 2025: {rho_2025:.3f}")

if __name__ == "__main__":
    main()
