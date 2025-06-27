"""Script to train a Voting MLP model using synthetic data."""

import torch
from torch.utils.data import DataLoader, random_split

from synth_data import SynthData
from voting_mlp import VotingMLP
from data_processing import DataProcessor
torch.manual_seed(42)

# Create data
num_samples = 10000 # TODO try different values
max_num_candidates = 5 # m
max_num_voters = 55 # n

data = SynthData(model_type="mlp")
dataset = data.generate_training_dataset(cand_max=max_num_candidates, vot_max=max_num_voters, num_samples=num_samples)

# Split dataset
train_data, (X_test, y_test) = data.split_data()

input_size = max_num_candidates * max_num_candidates * max_num_voters  # mmax² × nmax = 5² × 55 = 1375
model = VotingMLP(
    input_size=input_size,
    num_classes=max_num_candidates,
    train_loader=DataLoader(train_data, batch_size=200, shuffle=True)
)
# Train the model
model.train_model(num_gradient_steps=5000, seed=42) # TODO try different values

# Evaluate the model
model.evaluate_model(X_test, y_test)
