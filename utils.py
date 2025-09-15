import numpy as np
import pref_voting
import torch
from pref_voting.profiles import Profile



def encode_mlp(profile, cand_max=5, vot_max=55) -> torch.Tensor:
    """Encodes a profile for MLP models.

    Args:
        profile (pref_voting.profiles.Profile): The preference voting profile to be encoded.
        cand_max (int): Maximum number of candidates (alternatives).
        vot_max (int): Maximum number of voters.
    Returns:
        torch.Tensor: Encoded profile tensor of shape (1, cand_max * cand_max * vot_max).
    """
    X = np.zeros((1, cand_max * cand_max * vot_max), dtype=np.float32)

    encoded_profile = pad_profile(profile, mode="mlp")
    X[0] = encoded_profile

    return torch.tensor(X, dtype=torch.float32)


def encode_cnn(profile, cand_max=5, vot_max=55) -> torch.Tensor:
    """Encodes a profile for Convolutional Neural Network (CNN) models.

    Args:
        profile (pref_voting.profiles.Profile): The preference voting profile to be encoded.
        cand_max (int): Maximum number of candidates (alternatives).
        vot_max (int): Maximum number of voters.

    Returns:
        torch.utils.data.TensorDataset: Custom dataset containing the generated profiles and their winners.
    """
    X = np.zeros((1, cand_max, cand_max, vot_max), dtype=np.float32)

    encoded_profile = pad_profile(profile, mode="cnn")
    X[0] = encoded_profile

    return torch.tensor(X, dtype=torch.float32)


def pad_profile(profile: pref_voting.profiles.Profile, mode: str, cand_max: int = 5, vot_max: int = 55) -> np.ndarray:
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
