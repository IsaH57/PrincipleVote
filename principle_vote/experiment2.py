"""Script to run the experiments included in the project report"""
import json
import torch
from pref_voting.scoring_methods import borda
from torch.utils.data import DataLoader
import numpy as np
from pref_voting.generate_profiles import generate_profile

from principle_vote.axioms import check_anonymity, check_neutrality, check_condorcet, check_pareto, check_independence
from principle_vote.synth_data import SynthData
from principle_vote.voting_cnn import VotingCNN
from principle_vote.voting_mlp import VotingMLP
from principle_vote.voting_wec import VotingWEC

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Experiment 2: sample 400 profiles fulfilling a given axiom, train models with and without axiom loss on them
# Create data
train_samples = 15000
test_samples = 400
max_num_candidates = 5  # m
max_num_voters = 55  # n

def sample_profiles_fulfilling_axiom(axiom: str, num_samples: int, cand_max: int, vot_max: int, prob_model= "IC") -> SynthData:

    profiles = []
    winners = []
    while len(profiles)<num_samples:
        num_candidates = np.random.randint(1, cand_max + 1)
        num_voters = np.random.randint(1, vot_max + 1)
        prof = generate_profile(num_candidates, num_voters, probmodel=prob_model)
        winners = torch.tensor(borda(prof))


        if axiom == "anonymity":
            check = check_anonymity(prof, winners, cand_max)
        elif axiom == "neutrality":
            check = check_neutrality(prof, winners, cand_max)
        elif axiom == "condorcet":
            check = check_condorcet(prof, winners, cand_max)
        elif axiom == "pareto":
            check = check_pareto(prof, winners, cand_max)
        elif axiom == "independence":
            check= check_independence(prof, winners, cand_max)
        else:
            raise ValueError(f"Axiom '{axiom}' not recognized.")

        if check == 1:
            profiles.append(prof)
            winners.append(winners)
        else:
            continue

    return SynthData(cand_max=max_num_candidates, vot_max=max_num_voters, num_samples=num_samples,
                     prob_model="IC",
                     winner_method="borda").set_raw_data(profiles, winners)


all_axioms = ["anonymity", "neutrality", "condorcet", "pareto", "independence"]
results = {}

for a in all_axioms:

    print(a)
    data_train = SynthData(cand_max=max_num_candidates, vot_max=max_num_voters, num_samples=train_samples,
                             prob_model="IC",
                             winner_method="borda")
    data_test = sample_profiles_fulfilling_axiom(axiom=a, num_samples=test_samples, cand_max=max_num_candidates, vot_max=max_num_voters, prob_model="IC")

    ###########
    # MLP Models
    print("MLP")
    # Generate synthetic data for MLP
    mlp_dataset_train = data_train.encode_mlp()

    mlp_model = VotingMLP(
        train_loader=DataLoader(mlp_dataset_train, batch_size=200, shuffle=True,
                                collate_fn=data_train.collate_profile),
        max_candidates=max_num_candidates, max_voters=max_num_voters
    ).to(device)

    # Training
    mlp_model.train_model(num_steps=1000, seed=42, plot=True, axiom="none")

    # Evaluation
    data_test.encode_mlp()
    mlp_X_test, mlp_y_test = data_test.get_encoded_mlp()

    acc_hard_mlp = mlp_model.evaluate_model_hard(mlp_X_test, mlp_y_test)
    acc_soft_mlp = mlp_model.evaluate_model_soft(mlp_X_test, mlp_y_test)
    axiom_satisfaction_mlp = mlp_model.evaluate_axiom_satisfaction(data_test, axiom=a)

    # MLP for axiom
    mlp_model_a = VotingMLP(
        train_loader=DataLoader(mlp_dataset_train, batch_size=200, shuffle=True,
                                collate_fn=data_train.collate_profile),
        max_candidates=max_num_candidates, max_voters=max_num_voters
    ).to(device)

    # Training
    mlp_model_a.train_model(num_steps=1000, seed=42, plot=True, axiom=a)

    # Evaluation
    acc_hard_mlp_a = mlp_model_a.evaluate_model_hard(mlp_X_test, mlp_y_test)
    acc_soft_mlp_a = mlp_model_a.evaluate_model_soft(mlp_X_test, mlp_y_test)
    axiom_satisfactio_mlp_a = mlp_model_a.evaluate_axiom_satisfaction(data_test, axiom=a)

    ##########
    # CNN Model
    print("CNN")
    # Generate synthetic data for CNN
    cnn_dataset_train = data_train.encode_cnn()

    cnn_model = VotingCNN(train_loader=DataLoader(cnn_dataset_train, batch_size=200, shuffle=True,
                                              collate_fn=data_train.collate_profile),
                      max_candidates=max_num_candidates, max_voters=max_num_voters).to(device)

    # Training
    cnn_model.train_model(num_steps=5000, seed=42, plot=True, axiom="none")

    # Evaluation
    data_test.encode_cnn()
    cnn_X_test, cnn_y_test = data_test.get_encoded_cnn()
    acc_hard_cnn = cnn_model.evaluate_model_hard(cnn_X_test, cnn_y_test)
    acc_soft_cnn = cnn_model.evaluate_model_soft(cnn_X_test, cnn_y_test)
    axiom_satisfaction_cnn = cnn_model.evaluate_axiom_satisfaction(data_test, axiom=a)

    # CNN for Axiom
    cnn_model_a = VotingCNN(train_loader=DataLoader(cnn_dataset_train, batch_size=200, shuffle=True,
                                                  collate_fn=data_train.collate_profile),
                          max_candidates=max_num_candidates, max_voters=max_num_voters).to(device)

    # Training
    cnn_model_a.train_model(num_steps=5000, seed=42, plot=True, axiom=a)

    # Evaluation
    acc_hard_cnn_a = cnn_model_a.evaluate_model_hard(cnn_X_test, cnn_y_test)
    acc_soft_cnn_a = cnn_model_a.evaluate_model_soft(cnn_X_test, cnn_y_test)
    axiom_satisfaction_cnn_a = cnn_model_a.evaluate_axiom_satisfaction(data_test, axiom=a)

    ##########
    # WEC Models
    print("WEC")
    # Generate synthetic data for WEC
    wec_dataset = data_train.encode_wec()

    # Initialize WEC model
    wec_model = VotingWEC(max_candidates=max_num_candidates, max_voters=max_num_voters, corpus_size=2 * 10 ** 4,
                      embed_dim=100, window_size=5).to(device)

    # create DataLoader with custom collate function
    wec_train_loader = torch.utils.data.DataLoader(
        wec_dataset,
        batch_size=200,
        shuffle=True,
        collate_fn=data_train.collate_profile_wec
    )

    # Training
    wec_model.train_model(num_steps=5000, train_loader=wec_train_loader, seed=42, plot=True, axiom="none")

    # Evaluation
    data_test.encode_wec()
    wec_X_test, wec_y_test = data_test.get_encoded_wec()

    acc_hard_wec = wec_model.evaluate_model_hard(wec_X_test, wec_y_test)
    acc_soft_wec = wec_model.evaluate_model_soft(wec_X_test, wec_y_test)
    axiom_satisfaction_wec = wec_model.evaluate_axiom_satisfaction(data_test, axiom=a)

    # Initialize WEC model for Axioms
    wec_model_a = VotingWEC(max_candidates=max_num_candidates, max_voters=max_num_voters, corpus_size=2 * 10 ** 4,
                          embed_dim=100, window_size=5).to(device)

    # Training
    wec_model_a.train_model(num_steps=5000, train_loader=wec_train_loader, seed=42, plot=True, axiom=a)

    # Evaluation
    acc_hard_wec_a = wec_model.evaluate_model_hard(wec_X_test, wec_y_test)
    acc_soft_wec_a = wec_model.evaluate_model_soft(wec_X_test, wec_y_test)
    axiom_satisfaction_wec_a = wec_model_a.evaluate_axiom_satisfaction(data_test, axiom=a)


    #write to json
    results[a] = {
        "MLP": {
            "accuracy_hard": acc_hard_mlp,
            "accuracy_soft": acc_soft_mlp,
            "axiom_satisfaction": axiom_satisfaction_mlp,
            "accuracy_hard_axiom": acc_hard_mlp_a,
            "accuracy_soft_axiom": acc_soft_mlp_a,
            "axiom_satisfaction_axiom": axiom_satisfactio_mlp_a
        },
        "CNN": {
            "accuracy_hard": acc_hard_cnn,
            "accuracy_soft": acc_soft_cnn,
            "axiom_satisfaction": axiom_satisfaction_cnn,
            "accuracy_hard_axiom": acc_hard_cnn_a,
            "accuracy_soft_axiom": acc_soft_cnn_a,
            "axiom_satisfaction_axiom": axiom_satisfaction_cnn_a
        },
        "WEC": {
            "accuracy_hard": acc_hard_wec,
            "accuracy_soft": acc_soft_wec,
            "axiom_satisfaction": axiom_satisfaction_wec,
            "accuracy_hard_axiom": acc_hard_wec_a,
            "accuracy_soft_axiom": acc_soft_wec_a,
            "axiom_satisfaction_axiom": axiom_satisfaction_wec_a
        }
    }

# Write all results to the JSON file
with open("results/accuracies_axioms.json", "w") as f:
    json.dump(results, f, indent=4)