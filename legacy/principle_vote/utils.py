import numpy as np
import pref_voting
import torch
from pref_voting.profiles import Profile



def encode_mlp(profile, cand_max=5, vot_max=55, encoding_type="pairwise") -> torch.Tensor:
    """Encodes a profile for MLP models.

    Args:
        profile (pref_voting.profiles.Profile): The preference voting profile to be encoded.
        cand_max (int): Maximum number of candidates (alternatives).
        vot_max (int): Maximum number of voters.
        encoding_type (str): Type of encoding. Either "pairwise", "pairwise_per_voter", or "onehot". Defaults to "pairwise".
    Returns:
        torch.Tensor: Encoded profile tensor. Shape depends on encoding_type:
            - pairwise: (1, cand_max * (cand_max - 1) // 2) - symmetric pairwise scores [-1, 1]
            - pairwise_per_voter: (1, vot_max * cand_max * (cand_max - 1) // 2) - per-voter {-1,0,1}
            - onehot: (1, cand_max * cand_max * vot_max)
    """
    if encoding_type in ["pairwise", "pairwise_per_voter"]:
        input_size = cand_max * (cand_max - 1) // 2
        if encoding_type == "pairwise_per_voter":
            input_size *= vot_max
    else:  # onehot
        input_size = cand_max * cand_max * vot_max
    
    X = np.zeros((1, input_size), dtype=np.float32)

    encoded_profile = pad_profile(profile, mode="mlp", cand_max=cand_max, vot_max=vot_max, encoding_type=encoding_type)
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


def pad_profile(profile: pref_voting.profiles.Profile, mode: str, cand_max: int = 5, vot_max: int = 55, encoding_type: str = "pairwise") -> np.ndarray:
    """Padding logic for CNN and MLP.

    Args:
        profile (pref_voting.profiles.Profile): The preference voting profile to be padded.
        mode (str): The mode of encoding, either "mlp" or "cnn".
        cand_max (int): Maximum number of candidates (alternatives).
        vot_max (int): Maximum number of voters.
        encoding_type (str): Type of encoding for MLP. Either "pairwise" or "onehot". Defaults to "pairwise".

    Returns:
        np.ndarray
            - MLP (pairwise): flat vector with symmetric pairwise comparison matrix (upper triangle)
                  Entry for pair (i,j) = (voters preferring i over j - voters preferring j over i) / total
                  Values in range [-1, 1]. Shape: (cand_max * (cand_max - 1) // 2,)
            - MLP (onehot): flat vector length cand_max^2 * vot_max, Fortran order
            - CNN: (cand_max, cand_max, vot_max) one-hot [alt, rank, voter]
    """
    num_voters = profile.num_voters
    num_alternatives = profile.num_cands

    if num_voters > vot_max:
        raise ValueError(f"Number of voters ({num_voters}) exceeds maximum ({vot_max})")
    if num_alternatives > cand_max:
        raise ValueError(f"Number of alternatives ({num_alternatives}) exceeds maximum ({cand_max})")

    if mode == "mlp":
        if encoding_type == "pairwise":
            # Symmetric pairwise comparison matrix
            # Entry (i,j) = (votes preferring i over j - votes preferring j over i) / total_voters
            pairwise_matrix = np.zeros((cand_max, cand_max), dtype=np.float32)
            total_voters = sum(profile.counts)

            for ranking, count in zip(profile.rankings, profile.counts):
                for i, cand_i in enumerate(ranking):
                    if cand_i >= cand_max:
                        continue
                    for j, cand_j in enumerate(ranking):
                        if cand_j >= cand_max or cand_i == cand_j:
                            continue
                        if i < j:
                            pairwise_matrix[cand_i, cand_j] += count
                            pairwise_matrix[cand_j, cand_i] -= count

            if total_voters > 0:
                pairwise_matrix /= total_voters

            upper_triangle = []
            for i in range(cand_max):
                for j in range(i + 1, cand_max):
                    upper_triangle.append(pairwise_matrix[i, j])

            return np.array(upper_triangle, dtype=np.float32)
        if encoding_type == "pairwise_per_voter":
            # Per-voter preference embeddings: for each voter, encode upper triangle with {-1, 0, +1}
            tri_size = cand_max * (cand_max - 1) // 2
            encoded = np.zeros((vot_max, tri_size), dtype=np.float32)
            voter_idx = 0

            pair_indices = [(a, b) for a in range(cand_max) for b in range(a + 1, cand_max)]

            for ranking, count in zip(profile.rankings, profile.counts):
                truncated = tuple(ranking[:cand_max])
                positions = {cand: pos for pos, cand in enumerate(truncated)}
                default_pos = len(truncated)
                for _ in range(count):
                    if voter_idx >= vot_max:
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
