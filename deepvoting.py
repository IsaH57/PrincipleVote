"""Script to train a Voting MLP model using synthetic data."""
import numpy as np
from pref_voting.generate_profiles import generate_profile
# Experiment 1 with all 4 distributions:
# n=77, m=7, num_samples=15000
# Embeding: corpus size = 10^5, dim=200, window_size=7

# Experiment 2 and 3 with IC and Mallows:
# n=55, m=5,
# num_samples=5000 or 15000 (Experiment 3)
# Embeding: corpus size = 2*10^4, dim=100, window_size=5

import torch
from torch.utils.data import DataLoader, TensorDataset

from synth_data import SynthData
from voting_cnn import VotingCNN
from voting_mlp import VotingMLP
from voting_wec import VotingWEC

torch.manual_seed(42)

# Create data
num_samples = 15000
max_num_candidates = 5  # m
max_num_voters = 55  # n

data_train = SynthData(cand_max=max_num_candidates, vot_max=max_num_voters, num_samples=num_samples, prob_model="IC",
                       winner_method="borda")
data_test = SynthData(cand_max=max_num_candidates, vot_max=max_num_voters, num_samples=3000, prob_model="IC",
                      winner_method="borda")


#### MLP Model ####
print("MLP")
# Generate synthetic data for MLP
mlp_dataset_train = data_train.encode_mlp()

mlp_model = VotingMLP(
    train_loader=DataLoader(mlp_dataset_train, batch_size=200, shuffle=True, collate_fn=data_train.collate_profile),
    max_candidates=max_num_candidates, max_voters=max_num_voters
    )

# Training
mlp_model.train_model(num_steps=5000, seed=42, plot=True, axiom="independence")  # TODO try different values


# Evaluation
data_test.encode_mlp()
mlp_X_test, mlp_y_test = data_test.get_encoded_mlp()

mlp_model.evaluate_model_hard(mlp_X_test, mlp_y_test)
mlp_model.evaluate_model_soft(mlp_X_test, mlp_y_test)

# Prediction for a single example
mlp_single_example = mlp_X_test[0:1]
mlp_winner_mask_single, mlp_probs_single = mlp_model.predict(mlp_single_example)

print("\nMLP Prediction:")
print(f"Predicted Winner: {mlp_winner_mask_single.numpy()}")
print(f"Probabilities: {mlp_probs_single.numpy()}")


#### CNN Model ####
print("CNN")
# Generate synthetic data for CNN
cnn_dataset_train = data_train.encode_cnn()

cnn_model = VotingCNN(train_loader=DataLoader(cnn_dataset_train, batch_size=200, shuffle=True, collate_fn=data_train.collate_profile),
                      max_candidates=max_num_candidates, max_voters=max_num_voters)

# Training
cnn_model.train_model(num_steps=5000, seed=42, plot=True, axiom="independence")

# Evaluation
data_test.encode_cnn()
cnn_X_test, cnn_y_test = data_test.get_encoded_cnn()
cnn_model.evaluate_model_hard(cnn_X_test, cnn_y_test)
cnn_model.evaluate_model_soft(cnn_X_test, cnn_y_test)

# Single example prediction
cnn_single_example = cnn_X_test[0:1]
cnn_winner_mask_single, cnn_probs_single = cnn_model.predict(cnn_single_example)

print("\nCNN Prediction:")
print(f"Predicted Winner: {cnn_winner_mask_single.numpy()}")
print(f"Probabilities: {cnn_probs_single.numpy()}")



#### Embedding Classifier ####
print("WEC")
# Generate synthetic data for WEC
wec_dataset = data_train.encode_wec()

# Initialize WEC model
wec_model = VotingWEC(max_candidates=max_num_candidates, max_voters=max_num_voters, corpus_size=2 * 10 ** 4,
                      embed_dim=100, window_size=5)

# create DataLoader with custom collate function
wec_train_loader = torch.utils.data.DataLoader(
    wec_dataset,
    batch_size=200,
    shuffle=True,
    collate_fn=data_train.collate_profile_wec
)

# training
wec_model.train_model(num_steps=5000, train_loader=wec_train_loader, seed=42, plot=True, axiom="independence")


# evaluation
data_test.encode_wec()
wec_X_test, wec_y_test = data_test.get_encoded_wec()

wec_model.evaluate_model_hard(wec_X_test, wec_y_test)
wec_model.evaluate_model_soft(wec_X_test, wec_y_test)

# prediction
wec_single_example = wec_X_test[0:1]
wec_winner_mask_single, wec_probs_single = wec_model.predict(wec_single_example)

print("\nWEC Prediction:")
print(f"VPredicted Winner: {wec_winner_mask_single.numpy()}")
print(f"Probabilities: {wec_probs_single.numpy()}")
