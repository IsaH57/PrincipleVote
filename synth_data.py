"""SynthData: A Class for generating ad handling synthetic voting data."""

import numpy as np
import pref_voting
from pref_voting.generate_profiles import generate_profile
from pref_voting.scoring_methods import borda
import torch


class SynthData:
    def __init__(self, model_type: str = None, **kwargs):
        """ Initializes the SynthData class.

        Args:
            model_type (str, optional): Type of model to be used for encoding profiles. Defaults to None.

        Attributes:
            data (np.ndarray): Placeholder for synthetic voting data.
            model_type (str): Type of model to be used for encoding profiles.

        """
        self.data = None  # TODO store data here
        self.model_type = model_type

    def encode_pref_voting_profile_mlp(self, profile: pref_voting.profiles.Profile) -> np.ndarray:
        """Encodes a pref_voting.Profile object for use in an MLP.

        Args:
            profile (pref_voting.profiles.Profile): A pref_voting.Profile object.

        Returns:
            np.ndarray: Encoded profile as a NumPy array.
        """
        num_voters = profile.num_voters
        num_alternatives = profile.num_cands

        # One-hot encoding
        encoded_profile = np.zeros((num_voters, num_alternatives, num_alternatives))

        voter_idx = 0
        for ranking, count in zip(profile.rankings, profile.counts):
            for _ in range(count):
                for rank, alternative in enumerate(ranking):
                    encoded_profile[voter_idx, rank, alternative - 1] = 1
                voter_idx += 1

        return encoded_profile

    def generate_training_data(self, num_samples: int, num_candidates: int, num_voters: int,
                               winner_method: str = "borda") -> tuple:
        """Generates synthetic training data for a given model type.

        Args:
            num_samples (int): Number of samples to generate.
            num_candidates (int): Number of candidates in the voting profile.
            num_voters (int): Number of voters in the voting profile.
            model_type (str): Type of model for encoding (e.g., "mlp").
            winner_method (str): Method to compute the winner (e.g., "borda").

        Returns:
            tuple: A tuple containing the feature matrix X and the target vector y as PyTorch tensors.
        """
        X = []
        y = []

        for _ in range(num_samples):
            prof = generate_profile(num_candidates, num_voters,
                                    probmodel="IC")  # TODO: add more profile generation methods

            # Encode the profile based on the model type TODO: add more encoding methods
            if self.model_type == "mlp":
                encoded_profile = self.encode_pref_voting_profile_mlp(prof)
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")

            # Compute the winner using the specified method
            winner = borda(prof)
            if winner_method == "borda":
                winner = borda(prof)
            else:
                raise ValueError(f"Unsupported winner method: {winner_method}")

            # One-hot encoding of the winner
            winner_encoded = np.zeros(num_candidates)
            for w in winner:  # Assuming winner is a list of winning candidates TODO: change for other methods
                winner_encoded[w - 1] = 1

            X.append(encoded_profile)
            y.append(winner_encoded)

        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    def generate_training_dataset(self, num_samples: int, num_candidates: int, num_voters: int,
                                  winner_method: str = "borda") -> torch.utils.data.TensorDataset:
        """Generates a PyTorch dataset for training.

        Args:
            num_samples (int): Number of samples to generate.
            num_candidates (int): Number of candidates in the voting profile.
            num_voters (int): Number of voters in the voting profile.
            model_type (str): Type of model for encoding (e.g., "mlp").
            winner_method (str): Method to compute the winner (e.g., "borda").

        Returns:
            torch.utils.data.TensorDataset: A dataset containing the features and targets.
        """
        X, y = self.generate_training_data(num_samples, num_candidates, num_voters, winner_method)
        return torch.utils.data.TensorDataset(X, y)
