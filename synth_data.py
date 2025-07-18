"""SynthData: A Class for generating ad handling synthetic voting data."""
from typing import Any

import numpy as np
from pref_voting.generate_profiles import generate_profile
from pref_voting.scoring_methods import borda, plurality
import torch
from torch.utils.data import random_split, Subset

from data_processing import DataProcessor


class SynthData:
    SUPPORTED_MODELS = ["mlp", "cnn", "wec"]
    SUPPORTED_PROB_MODELS = ["IC", "MALLOWS-RELPHI", "Urn-R", "euclidean"]

    def __init__(self, model_type: str = None, **kwargs):
        """ Initializes the SynthData class.

        Args:
            model_type (str): Type of model to be used for encoding profiles.

        Attributes:
            data (torch.utils.data.TensorDataset): The generated dataset.
            model_type (str): The type of model used for encoding profiles.
            test_data (tuple): Tuple containing test data (X_test, y_test).

        """
        self.data = None
        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model type: {model_type}. Supported types are: {self.SUPPORTED_MODELS}")
        self.model_type = model_type
        self.test_data = None

    def generate_training_dataset_mlp(self, cand_max: int, vot_max: int, num_samples: int,
                                  prob_model: str = "IC",
                                  winner_method: str = "borda") -> torch.utils.data.TensorDataset:
        """Uses pref_voting to generate synthetic training data for MLP.

        Args:
            cand_max (int): Maximum number of alternatives.
            vot_max (int): Maximum number of voters.
            num_samples (int): Number of samples to generate.
            prob_model (str): Probability model for generating profiles.
            winner_method (str): Method to compute the winner.

        Returns:
            torch.utils.data.TensorDataset: Dataset containing the generated profiles and their winners.
        """
        if prob_model not in self.SUPPORTED_PROB_MODELS:
            raise ValueError(f"Unsupported probability model: {prob_model}")

        # Initialize np arrays with fixed shapes to hold the data
        X_np = np.zeros((num_samples, cand_max * cand_max * vot_max), dtype=np.float32)
        y_np = np.zeros((num_samples, cand_max), dtype=np.float32)

        # Generate data for each sample
        for i in range(num_samples):

            # Random number of candidates and voters for each sample TODO check if random is needed
            # num_candidates = np.random.randint(min_candidates, cand_max + 1)
            # num_voters = np.random.randint(min_voters, vot_max + 1)

            num_candidates = cand_max
            num_voters = vot_max

            # Generate a pref_voting profile
            prof = generate_profile(num_candidates, num_voters, probmodel=prob_model)

            data_processor = DataProcessor(prof)

            # Encode the profile based on the model type
            if self.model_type == "mlp":
                encoded_profile = data_processor.encode_pref_voting_profile_mlp(cand_max=cand_max, vot_max=vot_max)
            else:
                raise ValueError(f"Wrong model type: {self.model_type}. Use generate_training_dataset_{self.model_type} instead.")
            # Compute winner
            if winner_method == "borda":
                winner = borda(prof)
            elif winner_method == "plurality":
                winner = plurality(prof)
            else:
                raise ValueError(f"Unsupported winner method: {winner_method}")

            # Store data
            X_np[i] = encoded_profile
            # One-hot encode the winner
            for w in winner:
                y_np[i, w - 1] = 1

        self.data = torch.utils.data.TensorDataset(torch.tensor(X_np, dtype=torch.float32),
                                                   torch.tensor(y_np, dtype=torch.float32))

        return self.data

    def generate_training_dataset_cnn(self, cand_max: int, vot_max: int, num_samples: int,
                                        prob_model: str = "IC",
                                        winner_method: str = "borda") -> torch.utils.data.TensorDataset:
            """Generates synthetic training data for CNN models.

            Args:
                cand_max (int): Maximum number of alternatives.
                vot_max (int): Maximum number of voters.
                num_samples (int): Number of samples to generate.
                prob_model (str): Probability model for generating profiles.
                winner_method (str): Method to compute the winner.

            Returns:
                torch.utils.data.TensorDataset: Dataset containing the generated profiles and their winners.
            """
            if prob_model not in self.SUPPORTED_PROB_MODELS:
                raise ValueError(f"Unsupported probability model: {prob_model}")

            # Initialize np arrays with fixed shapes to hold the data
            X_np = np.zeros((num_samples, cand_max, cand_max, vot_max), dtype=np.float32)
            y_np = np.zeros((num_samples, cand_max), dtype=np.float32)

            # Generate data for each sample
            for i in range(num_samples):

                # Random number of candidates and voters for each sample TODO check if random is needed
                # num_candidates = np.random.randint(min_candidates, cand_max + 1)
                # num_voters = np.random.randint(min_voters, vot_max + 1)

                num_candidates = cand_max
                num_voters = vot_max

                # Generate a pref_voting profile
                prof = generate_profile(num_candidates, num_voters, probmodel=prob_model)

                data_processor = DataProcessor(prof)

                # Encode the profile based on the model type
                if self.model_type == "cnn":
                    encoded_profile = data_processor.encode_pref_voting_profile_cnn(cand_max=cand_max, vot_max=vot_max)
                else:
                    raise ValueError(f"Wrong model type: {self.model_type}. Use generate_training_dataset_{self.model_type} instead.")

                # Compute winner
                if winner_method == "borda":
                    winner = borda(prof)
                elif winner_method == "plurality":
                    winner = plurality(prof)
                else:
                    raise ValueError(f"Unsupported winner method: {winner_method}")

                # Store data
                X_np[i] = encoded_profile
                # One-hot encode the winner
                for w in winner:
                    y_np[i, w - 1] = 1
            self.data = torch.utils.data.TensorDataset(torch.tensor(X_np, dtype=torch.float32),
                                                         torch.tensor(y_np, dtype=torch.float32))
            return self.data

    def split_data(self, train_ratio: float = 0.8) -> tuple[Subset[Any], tuple[Any, Any]]:
        """Splits the dataset into training and test sets.

        Args:
            train_ratio (float): Proportion of the dataset to include in the training set.

        Returns:
            tuple[]: A tuple containing the training dataset and a tuple of test data (X_test, y_test).
        """
        if self.data is None:
            raise ValueError("Data has not been generated yet. Please call generate_training_dataset first.")

        train_size = int(len(self.data) * train_ratio)
        test_size = len(self.data) - train_size
        train_dataset, test_dataset = random_split(self.data, [train_size, test_size])
        self.train_data = self.data.tensors[0][train_dataset.indices], self.data.tensors[1][train_dataset.indices]
        self.test_data = self.data.tensors[0][test_dataset.indices], self.data.tensors[1][test_dataset.indices]
        return train_dataset, self.test_data


    def make_tensor(self, data: np.ndarray) -> torch.Tensor:
        """Converts a NumPy array to a PyTorch tensor."""
        return torch.tensor(data, dtype=torch.float32)
