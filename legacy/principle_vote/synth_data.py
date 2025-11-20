"""SynthData: A Class for generating ad handling synthetic voting data."""

import numpy as np
import pref_voting.profiles
from pref_voting.generate_profiles import generate_profile
from pref_voting.scoring_methods import borda, plurality
from pref_voting.c1_methods import copeland
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import random_split, Subset, Dataset, TensorDataset


class SynthData:
    """Class for generating and handling synthetic voting data.

        Attributes:
            cand_max (int): Maximum number of candidates (alternatives).
            vot_max (int): Maximum number of voters.
            num_samples (int): Number of samples to generate.
            prob_model (str): Probability model for generating profiles.
            winner_method (str): Method to compute the winner.
            samples (list): List of generated profiles.
            winners (np.ndarray): Array of winners corresponding to the generated profiles.
            mlp_encoded (np.ndarray): Encoded data for MLP models.
            cnn_encoded (np.ndarray): Encoded data for CNN models.
            mlp_data (TensorDataset): Dataset for MLP models.
            cnn_data (TensorDataset): Dataset for CNN models.
            wec_data (Dataset): Dataset for Word Embedding Classifier models.
    """
    SUPPORTED_MODELS = ["mlp", "cnn", "wec"]
    SUPPORTED_PROB_MODELS = ["IC", "MALLOWS-RELPHI", "Urn-R", "euclidean"]
    SUPPORTED_VOTING_RULES = ["borda", "plurality", "copeland"]

    def __init__(self, cand_max: int, vot_max: int, num_samples: int, prob_model: str, winner_method: str, encoding_type: str = "pairwise"):
        """Initializes the SynthData class used to generate synthetic training data for voting models.

        Args:
            cand_max (int): Maximum number of candidates (alternatives).
            vot_max (int): Maximum number of voters.
            num_samples (int): Number of samples to generate.
            prob_model (str): Probability model for generating profiles. Defaults to "IC".
            winner_method (str): Method to compute the winner. Defaults to "borda".
            encoding_type (str): Type of encoding for MLP. Either "pairwise", "pairwise_per_voter", or "onehot". Defaults to "pairwise".
        """
        # TODO make utils function
        if prob_model not in self.SUPPORTED_PROB_MODELS:
            raise ValueError(
                f"Unsupported probability model: {prob_model}. Supported models are: {self.SUPPORTED_PROB_MODELS}")
        if winner_method not in self.SUPPORTED_VOTING_RULES:
            raise ValueError(
                f"Unsupported winner method: {winner_method}. Supported methods are: {self.SUPPORTED_VOTING_RULES}")
        if encoding_type not in ["pairwise", "pairwise_per_voter", "onehot"]:
            raise ValueError(
                f"Unsupported encoding_type: {encoding_type}. Supported types are: ['pairwise', 'pairwise_per_voter', 'onehot']")

        self.cand_max = cand_max
        self.vot_max = vot_max
        self.num_samples = num_samples
        self.prob_model = prob_model
        self.winner_method = winner_method
        self.encoding_type = encoding_type

        self.samples = None
        self.winners = None

        self.mlp_encoded = None
        self.cnn_encoded = None

        self.mlp_data = None
        self.cnn_data = None
        self.wec_data = None

        self.generate_training_data(self.cand_max, self.vot_max, self.num_samples, self.prob_model,
                                       self.winner_method)

    def generate_training_data(self, cand_max: int, vot_max: int, num_samples: int,
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

        self.cand_max = cand_max
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
                winner = copeland(prof)
            else:
                raise ValueError(
                    f"Winner method not supported: {winner_method}. Supported methods are: {self.SUPPORTED_VOTING_RULES}")
            for w in winner:
                winners[i, w] = 1

        self.samples = samples
        self.winners = winners

        return self.samples, self.winners

    def encode_mlp(self) -> Dataset:
        """Encodes synthetic training data for Multi-Layer Perceptron (MLP) models.
        Supports aggregated pairwise, per-voter pairwise, and one-hot encodings.

        Returns:
            VotingDataset: Custom dataset containing the generated profiles and their winners.
        """
        # Calculate input size based on encoding type
        if self.encoding_type in ["pairwise", "pairwise_per_voter"]:
            # Pairwise: upper triangle of aggregated comparison matrix
            input_size = self.cand_max * (self.cand_max - 1) // 2
            if self.encoding_type == "pairwise_per_voter":
                input_size *= self.vot_max
        else:  # onehot
            input_size = self.cand_max * self.cand_max * self.vot_max
        
        X = np.zeros((self.num_samples, input_size), dtype=np.float32)

        for i, prof in enumerate(self.samples):
            encoded_profile = self.pad_profile(prof, mode="mlp")
            X[i] = encoded_profile

        self.mlp_encoded = X

        self.mlp_data = VotingDataset(
            self.samples,
            self.winners,
            X
        )

        return self.mlp_data

    def encode_cnn(self) -> Dataset:
        """Encodes synthetic training data for Convolutional Neural Network (CNN) models.

        Returns:
            VotingDataset: Custom dataset containing the generated profiles and their winners.
        """
        X = np.zeros((self.num_samples, self.cand_max, self.cand_max, self.vot_max), dtype=np.float32)

        for i, prof in enumerate(self.samples):
            encoded_profile = self.pad_profile(prof, mode="cnn")
            X[i] = encoded_profile

        self.cnn_encoded = X

        self.cnn_data = VotingDataset(
            self.samples,
            self.winners,
            X
        )

        return self.cnn_data

    def pad_profile(self, profile: pref_voting.profiles.Profile, mode: str) -> np.ndarray:
        """Padding logic for CNN and MLP.

        Args:
            profile (pref_voting.profiles.Profile): The preference voting profile to be padded.
            mode (str): The mode of encoding, either "mlp" or "cnn".

        Returns:
            np.ndarray
                    - MLP (pairwise): flat vector with symmetric pairwise comparison matrix (upper triangle)
                        Entry for pair (i,j) = (voters preferring i over j - voters preferring j over i) / total
                        Values in range [-1, 1]. Shape: (cand_max * (cand_max - 1) // 2,)
                    - MLP (pairwise_per_voter): concatenated per-voter upper triangles with values in {-1, 0, +1}
                        Shape: (vot_max * cand_max * (cand_max - 1) // 2,)
                - MLP (onehot): flat vector length cand_max^2 * vot_max, Fortran order
                - CNN: (cand_max, cand_max, vot_max) one-hot [alt, rank, voter]
        """
        num_voters = profile.num_voters
        num_alternatives = profile.num_cands

        if num_voters > self.vot_max:
            raise ValueError(f"Number of voters ({num_voters}) exceeds maximum ({self.vot_max})")
        if num_alternatives > self.cand_max:
            raise ValueError(f"Number of alternatives ({num_alternatives}) exceeds maximum ({self.cand_max})")

        if mode == "mlp":
            if self.encoding_type == "pairwise":
                # Symmetric pairwise comparison matrix with values in [-1, 1]
                pairwise_matrix = np.zeros((self.cand_max, self.cand_max), dtype=np.float32)
                total_voters = sum(profile.counts)

                for ranking, count in zip(profile.rankings, profile.counts):
                    for i, cand_i in enumerate(ranking):
                        if cand_i >= self.cand_max:
                            continue
                        for j, cand_j in enumerate(ranking):
                            if cand_j >= self.cand_max or cand_i == cand_j:
                                continue
                            if i < j:
                                pairwise_matrix[cand_i, cand_j] += count
                                pairwise_matrix[cand_j, cand_i] -= count

                if total_voters > 0:
                    pairwise_matrix /= total_voters

                upper_triangle = []
                for i in range(self.cand_max):
                    for j in range(i + 1, self.cand_max):
                        upper_triangle.append(pairwise_matrix[i, j])

                return np.array(upper_triangle, dtype=np.float32)
            if self.encoding_type == "pairwise_per_voter":
                tri_size = self.cand_max * (self.cand_max - 1) // 2
                encoded = np.zeros((self.vot_max, tri_size), dtype=np.float32)
                voter_idx = 0
                pair_indices = [(a, b) for a in range(self.cand_max) for b in range(a + 1, self.cand_max)]

                for ranking, count in zip(profile.rankings, profile.counts):
                    truncated = tuple(ranking[:self.cand_max])
                    positions = {cand: pos for pos, cand in enumerate(truncated)}
                    default_pos = len(truncated)
                    for _ in range(count):
                        if voter_idx >= self.vot_max:
                            break
                        vec = []
                        for a, b in pair_indices:
                            pos_a = positions.get(a, default_pos)
                            pos_b = positions.get(b, default_pos)
                            if pos_a == pos_b == default_pos:
                                vec.append(0.0)
                            elif pos_a == pos_b:
                                vec.append(0.0)
                            else:
                                vec.append(1.0 if pos_a < pos_b else -1.0)
                        encoded[voter_idx] = np.array(vec, dtype=np.float32)
                        voter_idx += 1

                return encoded.flatten(order="C")
            else:  # onehot encoding
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

    def set_raw_data(self,  profiles: list, winners: np.ndarray):
        """Sets raw profiles and winners directly.

        Args:
            profiles (list): List of Profile objects.
            winners (np.ndarray): Array of winners corresponding to the profiles.
        """
        self.samples = profiles
        self.winners = winners

    def get_encoded_mlp(self) -> tuple[torch.Tensor, torch.Tensor]:
        """ Gets the encoded MLP data as tensors.

        Returns:
            tuple: A tuple containing:
                - X (torch.Tensor): Encoded input tensor. Shape depends on encoding_type:
                    - pairwise: (num_samples, cand_max * (cand_max - 1) // 2) - symmetric scores [-1,1]
                    - pairwise_per_voter: (num_samples, vot_max * cand_max * (cand_max - 1) // 2) - per-voter {-1,0,1}
                    - onehot: (num_samples, cand_max * cand_max * vot_max)
                - y (torch.Tensor): Labels tensor of shape (num_samples, cand_max).
        """
        if self.mlp_encoded is None:
            raise ValueError("MLP data not encoded yet. Call encode_mlp() first.")
        return torch.tensor(self.mlp_encoded, dtype=torch.float32), torch.tensor(self.winners, dtype=torch.float32)

    def get_encoded_cnn(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Gets the encoded CNN data as tensors.

        Returns:
            tuple: A tuple containing:
                - X (torch.Tensor): Encoded input tensor of shape (num_samples, cand_max, cand_max, vot_max).
                - y (torch.Tensor): Labels tensor of shape (num_samples, cand_max).
        """
        if self.cnn_encoded is None:
            raise ValueError("CNN data not encoded yet. Call encode_cnn() first.")
        return torch.tensor(self.cnn_encoded, dtype=torch.float32), torch.tensor(self.winners, dtype=torch.float32)

    def get_encoded_wec(self) -> tuple[list, torch.Tensor]:
        """Gets the WEC data as profiles and tensor labels.

        Returns:
            tuple: A tuple containing:
                - profiles (list): List of Profile objects.
                - y (torch.Tensor): Labels tensor of shape (num_samples, cand_max).
        """
        if self.wec_data is None:
            raise ValueError("WEC data not encoded yet. Call encode_wec() first.")
        return self.samples, torch.tensor(self.winners, dtype=torch.float32)

    def get_raw_profiles(self) -> list:
        """Gets the raw profiles.

        Returns:
            list: List of Profile objects.
        """
        if self.samples is None:
            raise ValueError("No profiles found.")
        return self.samples

    def get_winners(self) -> np.ndarray:
        """Gets the winners.

        Returns:
            np.ndarray: Array of winners corresponding to the generated profiles.
        """
        if self.winners is None:
            raise ValueError("No winners found.")
        return self.winners

    def collate_profile(self, batch) -> tuple[list, torch.Tensor]:
        """Collate function for the DataLoader to process a batch of profiles and labels.

        Args:
            batch: List of tuples where each tuple contains a profile and its corresponding label.

        Returns:
            tuple: (profiles, labels) where profiles is a list of profiles and labels is a tensor of labels.
        """
        batch_x = torch.stack([torch.as_tensor(item[0]) for item in batch])
        batch_y = torch.stack([torch.as_tensor(item[1]) for item in batch])
        profiles = [item[2] for item in batch]

        return batch_x, batch_y, profiles

    def collate_profile_wec(self, batch) -> tuple[list, torch.Tensor]:
        """Collate function for the DataLoader of a WEC to process a batch of profiles and labels.

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

        # TODO add check if data exists
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


class VotingDataset(Dataset):
    """Custom Dataset for voting profiles and their corresponding labels."""

    def __init__(self, profiles, labels, encoded):
        self.profiles = profiles  # List of Profile objects
        self.labels = labels  # List of tensor labels
        self.x = encoded  # Function: Profile -> tensor

    def __len__(self):
        return len(self.profiles)

    def __getitem__(self, idx):
        profile = self.profiles[idx]
        x = self.x[idx]  # Encoded tensor for model
        y = self.labels[idx]  # Label tensor
        return x, y, profile  # <-- Now includes Profile object!
