"""Script to train a Voting MLP model using synthetic data."""

import torch
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader, random_split

from synth_data import SynthData
from voting_cnn import VotingCNN
from voting_mlp import VotingMLP
from data_processing import DataProcessor

torch.manual_seed(42)

# Create data
num_samples = 10000  # TODO try different values
max_num_candidates = 5  # m
max_num_voters = 55  # n

#### MLP Model ####
print("MLP")
# Generate synthetic data for MLP
mlp_data = SynthData(model_type="mlp")
mlp_dataset = mlp_data.generate_training_dataset_mlp(cand_max=max_num_candidates, vot_max=max_num_voters,
                                                     num_samples=num_samples)

# Split dataset
mlp_train_data, (mlp_X_test, mlp_y_test) = mlp_data.split_data()

input_size = max_num_candidates * max_num_candidates * max_num_voters  # mmax² × nmax = 5² × 55 = 1375
mlp_model = VotingMLP(
    input_size=input_size,
    num_classes=max_num_candidates,
    train_loader=DataLoader(mlp_train_data, batch_size=200, shuffle=True)
)
# Training
mlp_model.train_model(num_steps=5000, seed=42, plot=True)  # TODO try different values

# Evaluation
mlp_model.evaluate_model(mlp_X_test, mlp_y_test)



#### CNN Model ####
print("CNN")
# Generate synthetic data for CNN
cnn_data = SynthData(model_type="cnn")
cnn_dataset = cnn_data.generate_training_dataset_cnn(cand_max=max_num_candidates, vot_max=max_num_voters,
                                                     num_samples=num_samples)

# Split dataset
cnn_train_data, (cnn_X_test, cnn_y_test) = cnn_data.split_data()


cnn_model = VotingCNN(train_loader=DataLoader(cnn_train_data, batch_size=200, shuffle=True), m_max=5, n_max=55,
                      output_dim=5)
# Training
cnn_model.train_model(num_steps=5000, seed=42, plot=True)

# Evaluation
cnn_model.evaluate_model(cnn_X_test, cnn_y_test)
