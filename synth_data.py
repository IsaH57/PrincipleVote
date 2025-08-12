"""SynthData: A Class for generating ad handling synthetic voting data."""

import numpy as np
import pref_voting.profiles
from pref_voting.generate_profiles import generate_profile
from pref_voting.scoring_methods import borda, plurality
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import random_split, Subset, Dataset, TensorDataset


class SynthData:
    SUPPORTED_MODELS = ["mlp", "cnn", "wec"]
    SUPPORTED_PROB_MODELS = ["IC", "MALLOWS-RELPHI", "Urn-R", "euclidean"]
    SUPPORTED_VOTING_RULES = ["borda", "plurality", "copeland"]

    def __init__(self, cand_max: int, vot_max: int, num_samples: int, prob_model: str, winner_method: str):
        """Initializes the SynthData class.

        Args:
            cand_max (int): Maximum number of candidates (alternatives).
            vot_max (int): Maximum number of voters.
            num_samples (int): Number of samples to generate.
            prob_model (str): Probability model for generating profiles. Defaults to "IC".
            winner_method (str): Method to compute the winner. Defaults to "borda".

        Attributes:
            cand_max (int): Maximum number of candidates (alternatives).
            vot_max (int): Maximum number of voters.
            num_samples (int): Number of samples to generate.
            prob_model (str): Probability model for generating profiles.
            winner_method (str): Method to compute the winner.
            samples (list): List of generated profiles.
            winners (np.ndarray): Array of winners corresponding to the generated profiles.
            mlp_data (TensorDataset): Dataset for MLP models.
            cnn_data (TensorDataset): Dataset for CNN models.
            wec_data (Dataset): Dataset for Word Embedding Classifier models.
        """
        # TODO make utils function
        if prob_model not in self.SUPPORTED_PROB_MODELS:
            raise ValueError(
                f"Unsupported probability model: {prob_model}. Supported models are: {self.SUPPORTED_PROB_MODELS}")
        if winner_method not in self.SUPPORTED_VOTING_RULES:
            raise ValueError(
                f"Unsupported winner method: {winner_method}. Supported methods are: {self.SUPPORTED_VOTING_RULES}")

        self.cand_max = cand_max
        self.vot_max = vot_max
        self.num_samples = num_samples
        self.prob_model = prob_model
        self.winner_method = winner_method

        self.samples = None
        self.winners = None

        self.mlp_data = None
        self.cnn_data = None
        self.wec_data = None

        self.generate_training_dataset(self.cand_max, self.vot_max, self.num_samples, self.prob_model,
                                       self.winner_method)

    def generate_training_dataset(self, cand_max: int, vot_max: int, num_samples: int,
                                  prob_model: str = "IC", winner_method: str = "borda") -> tuple[list, np.ndarray]:
        """Generates synthetic training data for voting models.

        Args:
            cand_max (int): Maximum number of candidates (alternatives).
            vot_max (int): Maximum number of voters.
            num_samples (int): Number of samples to generate.
            prob_model (str): Probability model for generating profiles. Defaults to "IC".
            winner_method (str): Method to compute the winner. Defaults to "borda".

        Returns:
            tuple: A tuple containing the list of generated profiles and a numpy array of winners.
        """

        if prob_model not in self.SUPPORTED_PROB_MODELS:
            raise ValueError(
                f"Unsupported probability model: {prob_model}. Supported models are: {self.SUPPORTED_PROB_MODELS}")
        if winner_method not in self.SUPPORTED_VOTING_RULES:
            raise ValueError(
                f"Unsupported winner method: {winner_method}. Supported methods are: {self.SUPPORTED_VOTING_RULES}")

        self.cand_max = cand_max  # TODO maybe not here
        self.vot_max = vot_max
        self.num_samples = num_samples

        samples = []
        winners = np.zeros((num_samples, cand_max), dtype=np.float32)

        for i in range(num_samples):
            num_candidates = np.random.randint(1, cand_max + 1)
            num_voters = np.random.randint(1, vot_max + 1)
            prof = generate_profile(num_candidates, num_voters, probmodel=prob_model)

            samples.append(prof)

            if winner_method == "borda":
                winner = borda(prof)
            elif winner_method == "plurality":
                winner = plurality(prof)
            elif winner_method == "copeland":
                raise NotImplementedError("Copeland method is not implemented yet.")
            else:
                raise ValueError(
                    f"Winner method not supported: {winner_method}. Supported methods are: {self.SUPPORTED_VOTING_RULES}")
            for w in winner:
                winners[i, w - 1] = 1

        self.samples = samples
        self.winners = winners

        return self.samples, self.winners


    def encode_mlp(self) -> TensorDataset:
        """Encodes synthetic training data for Multi-Layer Perceptron (MLP) models.

        Returns:
            torch.utils.data.TensorDataset: Custom dataset containing the generated profiles and their winners.
        """
        X = np.zeros((self.num_samples, self.cand_max * self.cand_max * self.vot_max), dtype=np.float32)

        for i, prof in enumerate(self.samples):
            encoded_profile = self.pad_profile(prof, mode="mlp")
            X[i] = encoded_profile

        self.mlp_data = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(self.
                         winners, dtype=torch.float32)
        )

        return self.mlp_data

    def encode_cnn(self) -> TensorDataset:
        """Encodes synthetic training data for Convolutional Neural Network (CNN) models.

        Returns:
            torch.utils.data.TensorDataset: Custom dataset containing the generated profiles and their winners.
        """
        X = np.zeros((self.num_samples, self.cand_max, self.cand_max, self.vot_max), dtype=np.float32)

        for i, prof in enumerate(self.samples):
            encoded_profile = self.pad_profile(prof, mode="cnn")
            X[i] = encoded_profile

        self.cnn_data = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(self.winners, dtype=torch.float32)
        )

        return self.cnn_data

    def pad_profile(self, profile: pref_voting.profiles.Profile, mode: str):
        """Padding logic for CNN and MLP.

        Args:
            profile (pref_voting.profiles.Profile): The preference voting profile to be padded.
            mode (str): The mode of encoding, either "mlp" or "cnn".

        Returns:
            np.ndarray
                - MLP: flat vector length cand_max^2 * vot_max, Fortran order
                - CNN: (cand_max, cand_max, vot_max) one-hot [alt, rank, voter]
        """
        num_voters = profile.num_voters
        num_alternatives = profile.num_cands

        if num_voters > self.vot_max:
            raise ValueError(f"Number of voters ({num_voters}) exceeds maximum ({self.vot_max})")
        if num_alternatives > self.cand_max:
            raise ValueError(f"Number of alternatives ({num_alternatives}) exceeds maximum ({self.cand_max})")

        if mode == "mlp":
            encoded = np.zeros((self.cand_max, self.cand_max, self.vot_max), dtype=np.float32)
            voter_idx = 0
            for ranking, count in zip(profile.rankings, profile.counts):
                for _ in range(count):
                    if voter_idx < self.vot_max:
                        for rank_pos, alt in enumerate(ranking):
                            if rank_pos < self.cand_max:
                                if alt < self.cand_max:
                                    encoded[alt, rank_pos, voter_idx] = 1
                        voter_idx += 1
            return encoded.flatten(order='F')

        elif mode == "cnn":
            encoded = np.zeros((self.cand_max, self.cand_max, self.vot_max), dtype=np.float32)
            voter_idx = 0
            for ranking, count in zip(profile.rankings, profile.counts):
                for _ in range(count):
                    if voter_idx < self.vot_max:
                        for rank_pos, alt in enumerate(ranking):
                            if rank_pos < self.cand_max:
                                if alt < self.cand_max:
                                    encoded[alt, rank_pos, voter_idx] = 1
                        voter_idx += 1
            return encoded

        else:
            raise ValueError(f"Unknown mode: {mode}")

    def encode_wec(self) -> Dataset:
        """Encodes synthetic training data for Word Embedding Classifier (WEC) models.

        Returns:
            ProfileDataset: Custom dataset containing the generated profiles and their winners.
        """
        # Define a custom Dataset class for WEC
        class ProfileDataset(Dataset):
            def __init__(self, profiles, targets):
                self.profiles = profiles
                self.targets = targets

            def __len__(self):
                return len(self.profiles)

            def __getitem__(self, idx):
                return self.profiles[idx], self.targets[idx]

        self.wec_data = ProfileDataset(
            self.samples,
            torch.tensor(self.winners, dtype=torch.float32)
        )

        return self.wec_data

    def collate_profile(self, batch) -> tuple[list, torch.Tensor]:
        """Collate function for the DataLoader to process a batch of profiles and labels.

        Args:
            batch: List of tuples where each tuple contains a profile and its corresponding label.

        Returns:
            tuple: (profiles, labels) where profiles is a list of profiles and labels is a tensor of labels.
        """
        profiles = [item[0] for item in batch]
        labels = torch.stack([item[1] for item in batch])

        return profiles, labels

    def split_data(self, mode: str, train_ratio: float = 0.8) -> tuple[Dataset, tuple[list, torch.Tensor]]:
        """Splits the dataset into training and test sets.

        Args:
            mode (str): The mode of the dataset, either "mlp", "cnn", or "wec".
            train_ratio (float): Proportion of the dataset to include in the training set.

        Returns:
            tuple: A tuple containing the training dataset and a tuple of test data (X_test, y_test).
        """

        # TODO check if data is there
        # For TensorDatasets
        if mode == "mlp" or mode == "cnn":
            data = self.mlp_data if mode == "mlp" else self.cnn_data
            train_size = int(len(data) * train_ratio)
            test_size = len(data) - train_size
            train_dataset, test_dataset = random_split(data, [train_size, test_size])

            train_data = data.tensors[0][train_dataset.indices], data.tensors[1][train_dataset.indices]
            test_data = data.tensors[0][test_dataset.indices], data.tensors[1][test_dataset.indices]
        # For ProfileDataset
        elif mode == "wec":
            data = self.wec_data
            train_size = int(len(data) * train_ratio)
            test_size = len(data) - train_size
            indices = list(range(len(data)))

            # split indices into training and test sets
            train_indices, test_indices = train_test_split(indices, test_size=test_size, random_state=42)

            # create Subsets for training and test datasets
            train_dataset = Subset(data, train_indices)
            test_dataset = Subset(data, test_indices)

            # extract profiles and targets for test data
            X_test = [data.profiles[i] for i in test_indices]
            y_test = data.targets[test_indices]

            train_data = train_dataset
            test_data = (X_test, y_test)

        else:
            raise ValueError(f"Unsupported mode: {mode}. Supported modes are: {self.SUPPORTED_MODELS}.")

        return train_dataset, test_data
