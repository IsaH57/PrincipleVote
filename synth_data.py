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

    def __init__(self, model_type: str = None, **kwargs):
        """Initializes the SynthData class.

        Args:
            model_type (str): Type of model to be used for encoding profiles.

        Attributes:
            model_type (str): The type of model used for encoding profiles.
            data (torch.utils.data.TensorDataset): The generated dataset.
            train_data (tuple): Tuple containing training data (X_train, y_train).
            test_data (tuple): Tuple containing test data (X_test, y_test).
        """
        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model type: {model_type}. Supported types are: {self.SUPPORTED_MODELS}")

        self.model_type = model_type
        self.data = None
        self.train_data = None
        self.test_data = None

    # TODO make one function?
    def generate_training_dataset_mlp(self, cand_max, vot_max, num_samples, prob_model="IC",
                                      winner_method="borda") -> TensorDataset:
        """Generates synthetic training data for Multi-Layer Perceptron (MLP) models.

        Args:
            cand_max (int): Maximum number of candidates.
            vot_max (int): Maximum number of voters.
            num_samples (int): Number of samples to generate.
            prob_model (str): Probability model for generating profiles. Defaults to "IC".
            winner_method (str): Method to compute the winner. Defaults to "borda".

        Returns:
            torch.utils.data.TensorDataset: Custom dataset containing the generated profiles and their winners.
        """
        # TODO make utils function
        if prob_model not in self.SUPPORTED_PROB_MODELS:
            raise ValueError(
                f"Unsupported probability model: {prob_model}. Supported models are: {self.SUPPORTED_PROB_MODELS}")
        if winner_method not in self.SUPPORTED_VOTING_RULES:
            raise ValueError(
                f"Unsupported winner method: {winner_method}. Supported methods are: {self.SUPPORTED_VOTING_RULES}")

        X_np = np.zeros((num_samples, cand_max * cand_max * vot_max), dtype=np.float32)
        y_np = np.zeros((num_samples, cand_max), dtype=np.float32)

        for i in range(num_samples):
            num_candidates = np.random.randint(1, cand_max + 1)
            num_voters = np.random.randint(1, vot_max + 1)
            prof = generate_profile(num_candidates, num_voters, probmodel=prob_model)

            encoded_profile = self.pad_profile(prof, cand_max, vot_max, mode="mlp")
            X_np[i] = encoded_profile

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
                y_np[i, w - 1] = 1

        self.data = TensorDataset(
            torch.tensor(X_np, dtype=torch.float32),
            torch.tensor(y_np, dtype=torch.float32)
        )
        return self.data

    def generate_training_dataset_cnn(self, cand_max, vot_max, num_samples, prob_model="IC",
                                      winner_method="borda") -> TensorDataset:
        """Generates synthetic training data for Convolutional Neural Network (CNN) models.

        Args:
            cand_max (int): Maximum number of candidates.
            vot_max (int): Maximum number of voters.
            num_samples (int): Number of samples to generate.
            prob_model (str): Probability model for generating profiles. Defaults to "IC".
            winner_method (str): Method to compute the winner. Defaults to "borda".

        Returns:
            torch.utils.data.TensorDataset: Custom dataset containing the generated profiles and their winners.
        """
        # TODO make utils function
        if prob_model not in self.SUPPORTED_PROB_MODELS:
            raise ValueError(
                f"Unsupported probability model: {prob_model}. Supported models are: {self.SUPPORTED_PROB_MODELS}")
        if winner_method not in self.SUPPORTED_VOTING_RULES:
            raise ValueError(
                f"Unsupported winner method: {winner_method}. Supported methods are: {self.SUPPORTED_VOTING_RULES}")

        X_np = np.zeros((num_samples, cand_max, cand_max, vot_max), dtype=np.float32)
        y_np = np.zeros((num_samples, cand_max), dtype=np.float32)

        for i in range(num_samples):
            num_candidates = np.random.randint(1, cand_max + 1)
            num_voters = np.random.randint(1, vot_max + 1)
            prof = generate_profile(num_candidates, num_voters, probmodel=prob_model)

            encoded_profile = self.pad_profile(prof, cand_max, vot_max, mode="cnn")
            X_np[i] = encoded_profile

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
                y_np[i, w - 1] = 1

        self.data = TensorDataset(
            torch.tensor(X_np, dtype=torch.float32),
            torch.tensor(y_np, dtype=torch.float32)
        )
        return self.data

    def pad_profile(self, profile: pref_voting.profiles.Profile, cand_max: int, vot_max: int, mode: str):
        """Padding logic for CNN and MLP.

        Args:
            profile (pref_voting.profiles.Profile): The preference voting profile to be padded.
            cand_max (int): Maximum number of candidates (alternatives).
            vot_max (int): Maximum number of voters.
            mode (str): The mode of encoding, either "mlp" or "cnn".

        Returns:
            np.ndarray
                - MLP: flat vector length cand_max^2 * vot_max, Fortran order
                - CNN: (cand_max, cand_max, vot_max) one-hot [alt, rank, voter]
        """
        num_voters = profile.num_voters
        num_alternatives = profile.num_cands

        if num_voters > vot_max:
            raise ValueError(f"Number of voters ({num_voters}) exceeds maximum ({vot_max})")
        if num_alternatives > cand_max:
            raise ValueError(f"Number of alternatives ({num_alternatives}) exceeds maximum ({cand_max})")

        if mode == "mlp":
            encoded = np.zeros((cand_max, cand_max, vot_max), dtype=np.float32)
            voter_idx = 0
            for ranking, count in zip(profile.rankings, profile.counts):
                for _ in range(count):
                    if voter_idx < vot_max:
                        for rank_pos, alt in enumerate(ranking):
                            if rank_pos < cand_max:
                                if alt < cand_max:
                                    encoded[alt, rank_pos, voter_idx] = 1
                        voter_idx += 1
            return encoded.flatten(order='F')

        elif mode == "cnn":
            encoded = np.zeros((cand_max, cand_max, vot_max), dtype=np.float32)
            voter_idx = 0
            for ranking, count in zip(profile.rankings, profile.counts):
                for _ in range(count):
                    if voter_idx < vot_max:
                        for rank_pos, alt in enumerate(ranking):
                            if rank_pos < cand_max:
                                if alt < cand_max:
                                    encoded[alt, rank_pos, voter_idx] = 1
                        voter_idx += 1
            return encoded
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def generate_training_dataset_wec(self, cand_max: int, vot_max: int, num_samples: int,
                                      prob_model: str = "IC",
                                      winner_method: str = "borda") -> Dataset:
        """Generates synthetic training data for Word Embedding Classifier (WEC) models.

        Args:
            cand_max (int): Maximum number of candidates.
            vot_max (int): Maximum number of voters.
            num_samples (int): Number of samples to generate.
            prob_model (str): Probability model for generating profiles. Defaults to "IC".
            winner_method (str): Method to compute the winner. Defaults to "borda".

        Returns:
            ProfileDataset: Custom dataset containing the generated profiles and their winners.

        """
        if prob_model not in self.SUPPORTED_PROB_MODELS:
            raise ValueError(f"Unsupported probability model: {prob_model}")

        # Initialise np arrays with fixed shapes to hold the data
        X_profiles = []
        y_np = np.zeros((num_samples, cand_max), dtype=np.float32)

        # Generate data for each sample
        for i in range(num_samples):
            # Random number of candidates and voters for each sample
            num_candidates = np.random.randint(1, cand_max + 1)
            num_voters = np.random.randint(1, vot_max + 1)

            # Generate a pref_voting profile
            prof = generate_profile(num_candidates, num_voters, probmodel=prob_model)

            # save the profile
            X_profiles.append(prof)

            # get winner
            if winner_method == "borda":
                winner = borda(prof)
            elif winner_method == "plurality":
                winner = plurality(prof)
            else:
                raise ValueError(f"Winner method not supported: {winner_method}")

            # One-Hot ecode the winner
            for w in winner:
                y_np[i, w - 1] = 1

        # Define a custom Dataset class for WEC
        class ProfileDataset(Dataset):
            def __init__(self, profiles, targets):
                self.profiles = profiles
                self.targets = targets

            def __len__(self):
                return len(self.profiles)

            def __getitem__(self, idx):
                return self.profiles[idx], self.targets[idx]

        self.data = ProfileDataset(
            X_profiles,
            torch.tensor(y_np, dtype=torch.float32)
        )
        return self.data

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

    def split_data(self, train_ratio: float = 0.8) -> tuple[Dataset, tuple[list, torch.Tensor]]:
        """Splits the dataset into training and test sets.

        Args:
            train_ratio (float): Proportion of the dataset to include in the training set.

        Returns:
            tuple: A tuple containing the training dataset and a tuple of test data (X_test, y_test).
        """
        if self.data is None:
            raise ValueError("Data has not been generated yet. Please call generate_training_dataset first.")

        # For TensorDatasets
        if isinstance(self.data, torch.utils.data.TensorDataset):
            train_size = int(len(self.data) * train_ratio)
            test_size = len(self.data) - train_size
            train_dataset, test_dataset = random_split(self.data, [train_size, test_size])

            self.train_data = self.data.tensors[0][train_dataset.indices], self.data.tensors[1][train_dataset.indices]
            self.test_data = self.data.tensors[0][test_dataset.indices], self.data.tensors[1][test_dataset.indices]
        # For ProfileDataset
        else:
            train_size = int(len(self.data) * train_ratio)
            test_size = len(self.data) - train_size
            indices = list(range(len(self.data)))

            # split indices into training and test sets
            train_indices, test_indices = train_test_split(indices, test_size=test_size, random_state=42)

            # create Subsets for training and test datasets
            train_dataset = Subset(self.data, train_indices)
            test_dataset = Subset(self.data, test_indices)

            # extract profiles and targets for test data
            X_test = [self.data.profiles[i] for i in test_indices]
            y_test = self.data.targets[test_indices]

            self.train_data = train_dataset
            self.test_data = (X_test, y_test)

        return train_dataset, self.test_data
