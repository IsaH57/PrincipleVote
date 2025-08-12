"""Script to train a Voting MLP model using synthetic data."""

# Experiment 1 with all 4 distributions:
# n=77, m=7, num_samples=15000
#Embeding: corpus size = 10^5, dim=200, window_size=7

# Experiment 2 and 3 with IC and Mallows:
# n=55, m=5,
# num_samples=5000 or 15000 (Experiment 3)
#Embeding: corpus size = 2*10^4, dim=100, window_size=5

import torch
from torch.utils.data import DataLoader

from synth_data import SynthData
from voting_cnn import VotingCNN
from voting_mlp import VotingMLP
from voting_wec import VotingWEC

torch.manual_seed(42)

# Create data
num_samples = 15000
max_num_candidates = 5 # m
max_num_voters = 55  # n

data = SynthData(cand_max=max_num_candidates,vot_max=max_num_voters,num_samples=num_samples, prob_model="IC",winner_method="borda")


#### MLP Model ####

print("MLP")
# Generate synthetic data for MLP
mlp_dataset = data.encode_mlp()

# Split dataset
mlp_train_data, (mlp_X_test, mlp_y_test) = data.split_data(mode="mlp")

mlp_model = VotingMLP(train_loader=DataLoader(mlp_train_data, batch_size=200, shuffle=True), max_candidates=max_num_candidates, max_voters=max_num_voters
)
# Training
mlp_model.train_model(num_steps=5000, seed=42, plot=True)  # TODO try different values

# Evaluation
mlp_model.evaluate_model_hard(mlp_X_test, mlp_y_test)
mlp_model.evaluate_model_soft(mlp_X_test, mlp_y_test)

# Prediction
mlp_winner_mask, mlp_probabilities = mlp_model.predict(mlp_X_test)

# Prediction for a single example
mlp_single_example = mlp_X_test[0:1]
mlp_winner_mask_single, mlp_probs_single = mlp_model.predict(mlp_single_example)

print("\nMLP Prediction:")
print(f"Predicted Winner: {mlp_winner_mask_single.numpy()}")
print(f"Probabilities: {mlp_probs_single.numpy()}")


#### CNN Model ####
print("CNN")
# Generate synthetic data for CNN
cnn_dataset = data.encode_cnn()

# Split dataset
cnn_train_data, (cnn_X_test, cnn_y_test) = data.split_data(mode="cnn")


cnn_model = VotingCNN(train_loader=DataLoader(cnn_train_data, batch_size=200, shuffle=True), max_candidates=max_num_candidates, max_voters=max_num_voters)
# Training
cnn_model.train_model(num_steps=5000, seed=42, plot=True)

# Evaluation
cnn_model.evaluate_model_hard(cnn_X_test, cnn_y_test)
cnn_model.evaluate_model_soft(cnn_X_test, cnn_y_test)

# Prediction
winner_mask, probabilities = cnn_model.predict(cnn_X_test)

# Single example prediction
single_example = cnn_X_test[0:1]
winner_mask_single, probs_single = cnn_model.predict(single_example)

print("\nCNN Prediction:")
print(f"Predicted Winner: {winner_mask_single.numpy()}")
print(f"Probabilities: {probs_single.numpy()}")


#### Embedding Classifier ####

print("WEC")

# Generate synthetic data for WEC
wec_dataset = data.encode_wec()

# Split dataset
wec_train_data, (wec_X_test, wec_y_test) = data.split_data(mode="wec")

# Initialize WEC model
wec_model = VotingWEC(max_candidates=max_num_candidates, max_voters=max_num_voters, corpus_size=2 * 10 ** 4, embed_dim=100, window_size=5)



# create DataLoader with custom collate function
wec_train_loader = torch.utils.data.DataLoader(
    wec_train_data,
    batch_size=200,
    shuffle=True,
    collate_fn=data.collate_profile
)

# training
wec_model.train_model(num_steps=5000, train_loader=wec_train_loader, seed=42, plot=True)

# evaluation
wec_model.evaluate_model_hard(wec_X_test, wec_y_test)
wec_model.evaluate_model_soft(wec_X_test, wec_y_test)

# prediction
single_example = wec_X_test[0:1]
winner_mask, probs = wec_model.predict(single_example)

print("\nWEC Prediction:")
print(f"VPredicted Winner: {winner_mask.numpy()}")
print(f"Probabilities: {probs.numpy()}")
