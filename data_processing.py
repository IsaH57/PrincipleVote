"""DataProcessor: A class for preprocessing data to adjust it to different models."""

import numpy as np
import pref_voting
import torch


class DataProcessor:
    def __init__(self, profile: pref_voting.profiles.Profile):
        """Initializes the DataProcessor with a pref_voting.Profile object.

        Args:
            profile (pref_voting.profiles.Profile): A pref_voting.Profile object containing rankings and counts.

        Attributes:
            profile (pref_voting.profiles.Profile): The pref_voting.Profile object to be processed.
        """
        self.profile = profile

    def encode_pref_voting_profile_mlp(self,
                                       cand_max: int, vot_max: int) -> np.ndarray:
        """Encodes a pref_voting.Profile object for use in an MLP with padding and one-hot encoding.
        This method pads the profile to a maximum number of candidates (mmax) and voters (nmax),
        and encodes it into a flattened vector of dimension mmax² × nmax.

        Args:
            cand_max (int): Maximum number of candidate alternatives.
            vot_max (int): Maximum number of voters.

        Returns:
            np.ndarray: Flattened encoded profile of dimension mmax² × nmax.
        """
        num_voters = self.profile.num_voters
        num_alternatives = self.profile.num_cands

        if num_voters > vot_max:
            raise ValueError(f"Number of voters ({num_voters}) exceeds maximum ({vot_max})")
        if num_alternatives > cand_max:
            raise ValueError(f"Number of alternatives ({num_alternatives}) exceeds maximum ({cand_max})")

        # Initialize padded matrix prof with dimensions (mmax, mmax, nmax)
        # prof[rank, alternative, voter] = 1 if alternative is preferred at rank by voter
        prof = np.zeros((cand_max, cand_max, vot_max))

        voter_idx = 0
        for ranking, count in zip(self.profile.rankings, self.profile.counts):
            for _ in range(count):
                if voter_idx < vot_max:  # Only process voters within vot_max
                    for rank, alternative in enumerate(ranking):
                        if rank < cand_max:  # Only process ranks within cand_max
                            # One-hot encode alternative (alternative-1 because alternatives are 1-indexed)
                            alt_idx = alternative - 1
                            if alt_idx < cand_max:
                                prof[rank, alt_idx, voter_idx] = 1
                    voter_idx += 1

        # Flatten column by column to get vector x of dimension mmax² × nmax
        x = prof.flatten('F')  # Fortran-style flattening (column-major)

        return x