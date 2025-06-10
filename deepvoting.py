"""Script to train a Voting MLP model using synthetic data."""

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from synth_data import SynthData
from voting_mlp import VotingMLP

# Create data
num_samples = 500
num_candidates = 3
num_voters = 10
data = SynthData(model_type="mlp")
train_dataset = data.generate_training_dataset(num_samples=num_samples, num_candidates=num_candidates,
                                               num_voters=num_voters, winner_method="borda")
X_test, y_test = data.generate_training_data(num_samples=100, num_candidates=num_candidates, num_voters=num_voters,
                                             winner_method="borda")

# Create VotingMLP model
model = VotingMLP(input_shape=(num_voters, num_candidates, num_candidates), num_candidates=3)
model.set_train_loader(DataLoader(train_dataset, batch_size=32, shuffle=True))
model.set_criterion(nn.CrossEntropyLoss())
model.set_optimizer(optim.Adam(model.parameters(), lr=0.001))

# Train the model
model.train_model(num_epochs=10)

# Evaluate the model
model.evaluate_model(X_test, y_test)
