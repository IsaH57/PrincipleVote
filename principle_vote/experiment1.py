"""Script to run the experiments included in the project report"""
import json
import torch
from torch.utils.data import DataLoader

from principle_vote.synth_data import SynthData
from principle_vote.voting_cnn import VotingCNN
from principle_vote.voting_mlp import VotingMLP
from principle_vote.voting_wec import VotingWEC

torch.manual_seed(42)

# Experiment 1: get accuracies of all models on all rules
# Create data
train_samples = 15000
test_samples = 400
max_num_candidates = 5  # m
max_num_voters = 55  # n

print("borda")
data_train_borda = SynthData(cand_max=max_num_candidates, vot_max=max_num_voters, num_samples=train_samples,
                             prob_model="IC",
                             winner_method="borda")
data_test_borda = SynthData(cand_max=max_num_candidates, vot_max=max_num_voters, num_samples=test_samples,
                            prob_model="IC",
                            winner_method="borda")

###########
# MLP Model
print("MLP")
# Generate synthetic data for MLP
mlp_dataset_train = data_train_borda.encode_mlp()

mlp_model = VotingMLP(
    train_loader=DataLoader(mlp_dataset_train, batch_size=200, shuffle=True,
                            collate_fn=data_train_borda.collate_profile),
    max_candidates=max_num_candidates, max_voters=max_num_voters
)

# Training
mlp_model.train_model(num_steps=1000, seed=42, plot=True, axiom="none")

# Evaluation
data_test_borda.encode_mlp()
mlp_X_test, mlp_y_test = data_test_borda.get_encoded_mlp()

acc_hard_mlp_borda = mlp_model.evaluate_model_hard(mlp_X_test, mlp_y_test)
acc_soft_mlp_borda = mlp_model.evaluate_model_soft(mlp_X_test, mlp_y_test)

##########
# CNN Model
print("CNN")
# Generate synthetic data for CNN
cnn_dataset_train = data_train_borda.encode_cnn()

cnn_model = VotingCNN(train_loader=DataLoader(cnn_dataset_train, batch_size=200, shuffle=True,
                                              collate_fn=data_train_borda.collate_profile),
                      max_candidates=max_num_candidates, max_voters=max_num_voters)

# Training
cnn_model.train_model(num_steps=5000, seed=42, plot=True, axiom="none")

# Evaluation
data_test_borda.encode_cnn()
cnn_X_test, cnn_y_test = data_test_borda.get_encoded_cnn()
acc_hard_cnn_borda = cnn_model.evaluate_model_hard(cnn_X_test, cnn_y_test)
acc_soft_cnn_borda = cnn_model.evaluate_model_soft(cnn_X_test, cnn_y_test)

##########
# WEC Model
print("WEC")
# Generate synthetic data for WEC
wec_dataset = data_train_borda.encode_wec()

# Initialize WEC model
wec_model = VotingWEC(max_candidates=max_num_candidates, max_voters=max_num_voters, corpus_size=2 * 10 ** 4,
                      embed_dim=100, window_size=5)

# create DataLoader with custom collate function
wec_train_loader = torch.utils.data.DataLoader(
    wec_dataset,
    batch_size=200,
    shuffle=True,
    collate_fn=data_train_borda.collate_profile_wec
)

# Training
wec_model.train_model(num_steps=5000, train_loader=wec_train_loader, seed=42, plot=True, axiom="none")

# Evaluation
data_test_borda.encode_wec()
wec_X_test, wec_y_test = data_test_borda.get_encoded_wec()

acc_hard_wec_borda = wec_model.evaluate_model_hard(wec_X_test, wec_y_test)
acc_soft_wec_borda = wec_model.evaluate_model_soft(wec_X_test, wec_y_test)

####################
print("copeland")
data_train_copeland = SynthData(cand_max=max_num_candidates, vot_max=max_num_voters, num_samples=train_samples,
                                prob_model="IC",
                                winner_method="copeland")
data_test_copeland = SynthData(cand_max=max_num_candidates, vot_max=max_num_voters, num_samples=test_samples,
                               prob_model="IC",
                               winner_method="copeland")

###########
# MLP Model
print("MLP")
# Generate synthetic data for MLP
mlp_dataset_train = data_train_copeland.encode_mlp()

mlp_model = VotingMLP(
    train_loader=DataLoader(mlp_dataset_train, batch_size=200, shuffle=True,
                            collate_fn=data_train_copeland.collate_profile),
    max_candidates=max_num_candidates, max_voters=max_num_voters
)

# Training
mlp_model.train_model(num_steps=1000, seed=42, plot=True, axiom="none")

# Evaluation
data_test_copeland.encode_mlp()
mlp_X_test, mlp_y_test = data_test_copeland.get_encoded_mlp()

acc_hard_mlp_copeland = mlp_model.evaluate_model_hard(mlp_X_test, mlp_y_test)
acc_soft_mlp_copeland = mlp_model.evaluate_model_soft(mlp_X_test, mlp_y_test)

##########
# CNN Model
print("CNN")
# Generate synthetic data for CNN
cnn_dataset_train = data_train_copeland.encode_cnn()

cnn_model = VotingCNN(train_loader=DataLoader(cnn_dataset_train, batch_size=200, shuffle=True,
                                              collate_fn=data_train_copeland.collate_profile),
                      max_candidates=max_num_candidates, max_voters=max_num_voters)

# Training
cnn_model.train_model(num_steps=5000, seed=42, plot=True, axiom="none")

# Evaluation
data_test_copeland.encode_cnn()
cnn_X_test, cnn_y_test = data_test_copeland.get_encoded_cnn()
acc_hard_cnn_copeland = cnn_model.evaluate_model_hard(cnn_X_test, cnn_y_test)
acc_soft_cnn_copeland = cnn_model.evaluate_model_soft(cnn_X_test, cnn_y_test)

##########
# WEC Model
print("WEC")
# Generate synthetic data for WEC
wec_dataset = data_train_copeland.encode_wec()

# Initialize WEC model
wec_model = VotingWEC(max_candidates=max_num_candidates, max_voters=max_num_voters, corpus_size=2 * 10 ** 4,
                      embed_dim=100, window_size=5)

# create DataLoader with custom collate function
wec_train_loader = torch.utils.data.DataLoader(
    wec_dataset,
    batch_size=200,
    shuffle=True,
    collate_fn=data_train_copeland.collate_profile_wec
)

# Training
wec_model.train_model(num_steps=5000, train_loader=wec_train_loader, seed=42, plot=True, axiom="none")

# Evaluation
data_test_copeland.encode_wec()
wec_X_test, wec_y_test = data_test_copeland.get_encoded_wec()

acc_hard_wec_copeland = wec_model.evaluate_model_hard(wec_X_test, wec_y_test)
acc_soft_wec_copeland = wec_model.evaluate_model_soft(wec_X_test, wec_y_test)

####################
print("plurality")
data_train_plurality = SynthData(cand_max=max_num_candidates, vot_max=max_num_voters, num_samples=train_samples,
                                prob_model="IC",
                                winner_method="plurality")
data_test_plurality = SynthData(cand_max=max_num_candidates, vot_max=max_num_voters, num_samples=test_samples,
                               prob_model="IC",
                               winner_method="plurality")

###########
# MLP Model
print("MLP")
# Generate synthetic data for MLP
mlp_dataset_train = data_train_plurality.encode_mlp()

mlp_model = VotingMLP(
    train_loader=DataLoader(mlp_dataset_train, batch_size=200, shuffle=True,
                            collate_fn=data_train_plurality.collate_profile),
    max_candidates=max_num_candidates, max_voters=max_num_voters
)

# Training
mlp_model.train_model(num_steps=1000, seed=42, plot=True, axiom="none")

# Evaluation
data_test_plurality.encode_mlp()
mlp_X_test, mlp_y_test = data_test_plurality.get_encoded_mlp()

acc_hard_mlp_plurality = mlp_model.evaluate_model_hard(mlp_X_test, mlp_y_test)
acc_soft_mlp_plurality = mlp_model.evaluate_model_soft(mlp_X_test, mlp_y_test)

##########
# CNN Model
print("CNN")
# Generate synthetic data for CNN
cnn_dataset_train = data_train_plurality.encode_cnn()

cnn_model = VotingCNN(train_loader=DataLoader(cnn_dataset_train, batch_size=200, shuffle=True,
                                              collate_fn=data_train_plurality.collate_profile),
                      max_candidates=max_num_candidates, max_voters=max_num_voters)

# Training
cnn_model.train_model(num_steps=5000, seed=42, plot=True, axiom="none")

# Evaluation
data_test_plurality.encode_cnn()
cnn_X_test, cnn_y_test = data_test_plurality.get_encoded_cnn()
acc_hard_cnn_plurality= cnn_model.evaluate_model_hard(cnn_X_test, cnn_y_test)
acc_soft_cnn_plurality = cnn_model.evaluate_model_soft(cnn_X_test, cnn_y_test)

##########
# WEC Model
print("WEC")
# Generate synthetic data for WEC
wec_dataset = data_train_plurality.encode_wec()

# Initialize WEC model
wec_model = VotingWEC(max_candidates=max_num_candidates, max_voters=max_num_voters, corpus_size=2 * 10 ** 4,
                      embed_dim=100, window_size=5)

# create DataLoader with custom collate function
wec_train_loader = torch.utils.data.DataLoader(
    wec_dataset,
    batch_size=200,
    shuffle=True,
    collate_fn=data_train_plurality.collate_profile_wec
)

# Training
wec_model.train_model(num_steps=5000, train_loader=wec_train_loader, seed=42, plot=True, axiom="none")

# Evaluation
data_test_plurality.encode_wec()
wec_X_test, wec_y_test = data_test_plurality.get_encoded_wec()

acc_hard_wec_plurality = wec_model.evaluate_model_hard(wec_X_test, wec_y_test)
acc_soft_wec_plurality = wec_model.evaluate_model_soft(wec_X_test, wec_y_test)


# write to json
results = {
    "borda": {
        "mlp_borda": {
            "hard_accuracy": acc_hard_mlp_borda,
            "soft_accuracy": acc_soft_mlp_borda
        },
        "cnn_borda": {
            "hard_accuracy": acc_hard_cnn_borda,
            "soft_accuracy": acc_soft_cnn_borda
        },
        "wec_borda": {
            "hard_accuracy": acc_hard_wec_borda,
            "soft_accuracy": acc_soft_wec_borda
        }
    },
    "copeland": {
        "mlp_copeland": {
            "hard_accuracy": acc_hard_mlp_copeland,
            "soft_accuracy": acc_soft_mlp_copeland
        },
        "cnn_copeland": {
            "hard_accuracy": acc_hard_cnn_copeland,
            "soft_accuracy": acc_soft_cnn_copeland
        },
        "wec_copeland": {
            "hard_accuracy": acc_hard_wec_copeland,
            "soft_accuracy": acc_soft_wec_copeland
        }
    },
    "plurality": {
        "mlp_plurality": {
            "hard_accuracy": acc_hard_mlp_plurality,
            "soft_accuracy": acc_soft_mlp_plurality
        },
        "cnn_plurality": {
            "hard_accuracy": acc_hard_cnn_plurality,
            "soft_accuracy": acc_soft_cnn_plurality
        },
        "wec_plurality": {
            "hard_accuracy": acc_hard_wec_plurality,
            "soft_accuracy": acc_soft_wec_plurality
        }
    }
}
with open('results/accuracies_5.json', 'w') as f:
    json.dump(results, f, indent=4)
