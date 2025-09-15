"""Implementation of loss functions corresponding to various voting axioms."""
import torch
import torch.nn.functional as F
import random
import pref_voting
from pref_voting.other_methods import pareto
from random import sample
from pref_voting.profiles import Profile
from torch import Tensor

import utils


#### Anonymity Loss ####
def permute_voters_mlp(X_batch, cand_max, vot_max) -> torch.Tensor:
    """Permute voter dimension for MLP input.

    Args:
        X_batch: tensor of shape (batch, cand^2 * vot)
        cand_max: maximum number of candidates
        vot_max: maximum number of voters

    Returns:
        torch.Tensor: Tensor of the same shape with voters permuted.
    """
    batch_size = X_batch.shape[0]

    # reshape back to (batch, cand, cand, vot)
    X_reshaped = X_batch.view(batch_size, cand_max, cand_max, vot_max)

    # generate a random voter permutation
    perm = torch.randperm(vot_max)

    # apply permutation along voter axis
    X_perm = X_reshaped[:, :, :, perm]

    # flatten back to (batch, cand^2 * vot) with Fortran order
    X_perm = X_perm.permute(0, 3, 1, 2).contiguous().view(batch_size, -1).contiguous()

    return X_perm


def permute_voters_cnn(X_batch, cand_max, vot_max) -> torch.Tensor:
    """Permute voter dimension for CNN input.

    Args:
        X_batch: tensor of shape (batch, cand, cand, voters)
        cand_max: maximum number of candidates
        vot_max: maximum number of voters

    Returns:
        torch.Tensor: Tensor of the same shape with voters permuted.
    """
    batch_size, cand, _, voters = X_batch.shape
    perm = torch.randperm(voters, device=X_batch.device)
    return X_batch[:, :, :, perm]


def permute_voters_wec(profiles, cand_max, vot_max) -> list[pref_voting.profiles.Profile]:
    """Permute voters in a list of pref_voting Profile objects.

    Args:
        profiles: list of pref_voting.profiles.Profile
        cand_max: maximum number of candidates
        vot_max: maximum number of voters

    Returns:
        list[pref_voting.profiles.Profile]: profiles with permuted voters
    """
    new_profiles = []

    for prof in profiles:
        # create a list with each ranking repeated according to its count
        expanded_rankings = []
        for ranking, count in zip(prof.rankings, prof.counts):
            expanded_rankings.extend([ranking] * count)

        # shuffle the list to permute voters
        random.shuffle(expanded_rankings)

        aggregated_rankings = []
        aggregated_counts = []

        while expanded_rankings:
            ranking = expanded_rankings[0]
            count = expanded_rankings.count(ranking)
            aggregated_rankings.append(ranking)
            aggregated_counts.append(count)
            expanded_rankings = [r for r in expanded_rankings if r != ranking]

        # create new Profiles with aggregated rankings and counts
        new_prof = pref_voting.profiles.Profile(aggregated_rankings, aggregated_counts)
        new_profiles.append(new_prof)

    return new_profiles

PERMUTATION_FUNCTIONS_VOT = {
    "mlp": permute_voters_mlp,
    "cnn": permute_voters_cnn,
    "wec": permute_voters_wec, }

def anonymity_loss(model, X_batch, num_samples=50, eps=1e-8) -> torch.Tensor:
    """Anonymity Loss: predictions should not depend on voter order.

    Args:
        model: the neural network model
        X_batch: tensor (batch, input_size) for MLP, batch, cand, cand, vot) for CNN or list of Profile for WEC
        num_samples: number of random voter permutations
        eps: small constant to avoid log(0)

    Returns:
        torch.Tensor: loss as average KL divergence between original and permuted predictions
    """
    original_logits = model(X_batch)
    original_probs = F.softmax(original_logits, dim=1)
    original_probs = torch.clamp(original_probs, min=eps, max=1.0)

    permute_func = PERMUTATION_FUNCTIONS_VOT.get(model.name)
    if permute_func is None:
        raise ValueError(f"Model '{model.name}' not supported")

    loss = 0.0
    for _ in range(num_samples):
        X_permuted = permute_func(X_batch, model.max_cand, model.max_vot)
        permuted_logits = model(X_permuted)
        permuted_probs = F.softmax(permuted_logits, dim=1)
        permuted_probs = torch.clamp(permuted_probs, min=eps, max=1.0)

        # KL divergence
        kl = F.kl_div(
            (original_probs + eps),
            permuted_probs,
            reduction="batchmean"
        )
        loss += kl

    return loss / num_samples


#### Neutrality Loss ####
def permute_candidates_mlp(X_batch, max_cand, max_vot, perm) -> torch.Tensor:
    """Permutes candidate dimensions for MLP input.

    Args:
        X_batch: tensor of shape (batch, cand^2 * vot)
        max_cand: maximum number of candidates
        max_vot: maximum number of voters

    Returns:
        torch.Tensor: Tensor of the same shape with candidates permuted.
   """

    batch_size = X_batch.size(0)
    X_perm = torch.zeros_like(X_batch)

    for i in range(batch_size):
        profile = X_batch[i].view(max_vot, max_cand, max_cand)
        permuted_profile = profile[:, perm][:, :, perm]  # permute rows+cols
        X_perm[i] = permuted_profile.flatten()

    return X_perm


def permute_candidates_cnn(X_batch, max_cand, max_vot, perm) -> torch.Tensor:
    """Permutes candidate dimensions for CNN input.

    Args:
        X_batch: tensor of shape (batch, cand, cand, voters)
        perm: tensor of shape (cand,) with a permutation of candidate indices

    Returns:
        torch.Tensor: Tensor of the same shape with candidates permuted.
    """
    batch_size, cand, _, voters = X_batch.shape
    X_perm = torch.zeros_like(X_batch)

    for i in range(batch_size):
        profile = X_batch[i]  # shape (cand, cand, voters)
        permuted_profile = profile[perm][:, perm]  # permute rows+cols
        X_perm[i] = permuted_profile

    return X_perm


def permute_candidates_wec(profiles, cand_max, vot_max, perm) -> list[pref_voting.profiles.Profile]:
    """Permute candidates in a list of pref_voting Profile objects using a given permutation.

    Args:
        profiles: list of pref_voting.profiles.Profile
        cand_max: maximum number of candidates
        vot_max: maximum number of voters
        perm: torch.Tensor of size (cand_max,) defining the candidate permutation

    Returns:
        list[pref_voting.profiles.Profile]: profiles with permuted candidates
    """
    new_profiles = []

    for prof in profiles:
        candidates = list(prof.candidates)  # candidate IDs
        n = len(candidates)
        perm = torch.randperm(n)

        # create a mapping from old to new candidate IDs based on permutation
        permutation_dict = {}
        for i, cand in enumerate(candidates):
            if i < len(perm):
                target_idx = perm[i].item() % n
                permutation_dict[cand] = candidates[target_idx]

        new_profiles.append(prof.apply_cand_permutation(permutation_dict))
    return new_profiles

PERMUTATION_FUNCTIONS_CAND = {
    "mlp": permute_candidates_mlp,
    "cnn": permute_candidates_cnn,
    "wec": permute_candidates_wec,
}

def neutrality_loss(model, X_batch, max_cand, max_vot, num_samples=50) -> torch.Tensor:
    """Neutrality Loss: enforces that permuting candidates in the profile corresponds to permuting them in the output distribution.

    Args:
        model: the neural network model
        X_batch: tensor (batch, input_size) for MLP, (batch, cand, cand, vot) for CNN or list of Profile for WEC
        max_cand: maximum number of candidates
        max_vot: maximum number of voters
        num_samples: number of random candidate permutations

    Returns:
        torch.Tensor: loss as average KL divergence between original and permuted predictions
    """
    original_logits = model(X_batch)

    permute_func = PERMUTATION_FUNCTIONS_CAND.get(model.name)
    if permute_func is None:
        raise ValueError(f"Model '{model.name}' not supported")

    loss = torch.zeros(1, device=original_logits.device).squeeze()

    for _ in range(num_samples):
        perm = torch.randperm(max_cand, device=original_logits.device)

        # first permuting then voting
        X_permuted = permute_func(X_batch, max_cand, max_vot, perm)
        logits_perm = model(X_permuted)
        log_probs_perm = F.log_softmax(logits_perm, dim=1)

        # first voting then permuting
        logits_orig = original_logits[:, perm]
        log_probs_orig_perm = F.log_softmax(logits_orig, dim=1)

        # KL divergence between the two distributions
        kl = F.kl_div(
            log_probs_perm,
            log_probs_orig_perm,
            reduction="batchmean",
            log_target=True
        )
        loss += kl

    return loss / num_samples


#### Condorcet Loss ####
def condorcet_loss(model, X_batch, prof, max_cand, max_vot) -> torch.Tensor:
    """Condorcet Loss: if a Condorcet winner exists, it should be chosen with high probability.

    Args:
        model: the neural network model
        X_batch: tensor (batch, input_size) for MLP, (batch, cand, cand, vot) for CNN or list of Profile for WEC
        max_cand: maximum number of candidates
        max_vot: maximum number of voters

    Returns:
        torch.Tensor: average loss penalizing probability of not choosing the Condorcet winner
    """
    logits = model(X_batch)
    probs = F.softmax(logits, dim=1)  # (batch, cand)
    loss = 0.0

    # MLP and CNN use tensors
    if model.name in ["mlp", "cnn"]:
        batch_size = X_batch.size(0)

        for i in range(batch_size):
            cw = prof[i].condorcet_winner()

            if cw is not None:
                loss += 1.0 - probs[i, cw]

        return loss / batch_size

    # WEC uses list of Profile objects
    elif model.name == "wec":
        batch_size = len(X_batch)

        for i, p in enumerate(X_batch):
            cw = p.condorcet_winner()
            if cw is not None:
                loss += 1.0 - probs[i, cw]

        return loss / batch_size

    else:
        raise ValueError(f"Unsupported model type: {model.name}")


#### Pareto Loss ####
def pareto_dominated(profile: pref_voting.profiles.Profile, max_cand):
    """Return list of candidates that are Pareto dominated in WEC profile.

    Args:
        profile: pref_voting Profile object
        max_cand: maximum number of candidates

    Returns:
        set: set of indices of Pareto dominated candidates
    """
    not_dominated = pareto(profile)

    all_candidates = set(profile.candidates)
    dominated = all_candidates - set(not_dominated)

    return dominated


def pareto_loss(model, X_batch, prof, max_cand, max_vot) -> torch.Tensor:
    """Pareto Loss: candidates that are Pareto dominated should not be chosen.

    Args:
        model: the neural network model
        X_batch: tensor (batch, input_size) for MLP, (batch, cand, cand, vot) for CNN or list of Profile for WEC
        max_cand: maximum number of candidates
        max_vot: maximum number of voters

    Returns:
        torch.Tensor: average loss penalizing probability of choosing Pareto dominated candidates
    """
    logits = model(X_batch)
    probs = F.softmax(logits, dim=1)
    loss = 0.0

    if model.name in ["mlp", "cnn"]:
        batch_size = X_batch.size(0)

        for i in range(batch_size):
            dominated = pareto_dominated(prof[i], max_cand)

            for b in dominated:
                loss += probs[i, b]

        return loss / batch_size

    elif model.name == "wec":
        batch_size = len(X_batch)

        for i, p in enumerate(X_batch):
            dominated = pareto_dominated(p, max_cand)
            for b in dominated:
                loss += probs[i, b]

        return loss / batch_size

    else:
        raise ValueError(f"Unsupported model type: {model.name}")


#### Independence  Loss ####
def independence_loss(model, X_batch, prof, max_cand, max_vot, num_samples=50) -> torch.Tensor:
    """Independence Loss: the relative probabilities of two alternatives a and b should not depend on other alternatives.

    Args:
        model: the neural network model
        X_batch: tensor (batch, input_size) for MLP, (batch, cand, cand, vot) for CNN or list of Profile for WEC
        prof: list of pref_voting Profile objects (needed to sample pairs of alternatives)
        max_cand: maximum number of candidates
        max_vot: maximum number of voters
        num_samples: number of random pairs of alternatives to sample
    """
    # only consider nontrivial profiles, i.e., with at least 2 alternatives
    X_nontrivial_p = [p for p in prof if p.num_cands > 1]
    if model.name in ["mlp", "cnn"]:
        X_nontrivial = torch.stack([batch for batch, p in zip(X_batch, prof) if p.num_cands > 1])
    elif model.name == "wec":
        X_nontrivial = [p for p in X_batch if p.num_cands > 1]

    # Compute prediction of model
    original_prediction = model(X_nontrivial)

    # initialize the loss
    loss = torch.zeros(1).squeeze()

    for _ in range(num_samples):
        # For each original nontrivial profile generate a permuted version
        X_permuted = []

        # whose rankings agree with the corresponding original ones in the
        # order of a given pair (a,b) of alternatives

        pairs_of_alternatives = []
        for p in X_nontrivial_p:
            # Choose two distinct alternatives a and b in prof
            pair_of_alternatives = sample(p.candidates, 2)
            a = pair_of_alternatives[0]
            b = pair_of_alternatives[1]
            pairs_of_alternatives.append((a, b))

            # now build permuted version of prof by randomly sampling rankings
            # that, however, must agree in the order of a and b with the
            # corresponding ranking in the original prof
            permuted_prof = []
            for ranking in p.rankings:
                while True:  # we'll break this while loop eventually (see below)
                    # make a copy of the original ranking
                    permuted_ranking = list(ranking)
                    # shuffle the copy
                    random.shuffle(permuted_ranking)
                    # check if it still agrees in the order of a and b
                    # with original ranking
                    if (permuted_ranking.index(a) > permuted_ranking.index(b)) == (
                            (ranking.index(a) > ranking.index(b))):
                        # if so, add it to permuted list of profiles and break
                        permuted_prof.append(permuted_ranking)
                        break
                    # we will break the while loop: there is a 50% chance that
                    # permuted_ranking agrees in the order of a and b with
                    # the original ranking

            if model.name == "mlp":
                permuted_prof_enc = utils.encode_mlp(Profile(permuted_prof), max_cand, max_vot)
                X_permuted.append(permuted_prof_enc)
            elif model.name == "cnn":
                permuted_prof_enc = utils.encode_cnn(Profile(permuted_prof), max_cand, max_vot)
                X_permuted.append(permuted_prof_enc)
            elif model.name == "wec":
                x_permuted = Profile(permuted_prof)
                X_permuted.append(x_permuted)
            else:
                raise ValueError(f"Unsupported model type: {model.name}")


        if model.name == "mlp":
            X_permuted = torch.stack(X_permuted, dim=0)
            # Next compute prediction of the model on the permuted profiles
            permuted_prediction = model(X_permuted).squeeze(1)

            on_the_one_hand = torch.stack(
                [original_prediction[i][[a, b]] for i, (a, b) in enumerate(pairs_of_alternatives)],
                dim=0
            )
            on_the_other_hand = torch.stack(
                [permuted_prediction[i][[a, b]] for i, (a, b) in enumerate(pairs_of_alternatives)],
                dim=0
            )

        elif model.name == "cnn":
            X_permuted = torch.stack(X_permuted, dim=0)
            # Next compute prediction of the model on the permuted profiles
            permuted_prediction = model(X_permuted.squeeze(1))

            on_the_one_hand = torch.stack(
                [original_prediction[i][[a, b]] for i, (a, b) in enumerate(pairs_of_alternatives)],
                dim=0
            )
            on_the_other_hand = torch.stack(
                [permuted_prediction[i][[a, b]] for i, (a, b) in enumerate(pairs_of_alternatives)],
                dim=0
            )

        elif model.name == "wec":
            on_the_one_hand = torch.stack(
                [original_prediction[i][pairs_of_alternatives[i],] for i in range(len(original_prediction))],
                dim=0
            )
            on_the_other_hand = torch.stack(
                [permuted_prediction[i][pairs_of_alternatives[i],] for i in range(len(permuted_prediction))],
                dim=0
            )

        loss += F.kl_div(
            F.softmax(on_the_one_hand, dim=1),
            F.softmax(on_the_other_hand, dim=1),
            reduction="batchmean"
        )
    # return average loss
    return (1 / num_samples) * loss


def set_training_axiom(model, batch_X, prof, axiom: str, lambda_axiom: float = 1e-6) -> int | Tensor:
    """Selects and computes the loss corresponding to the specified axiom.

    Args:
        model: the neural network model
        batch_X: tensor (batch, input_size) for MLP, (batch, cand, cand, vot) for CNN or list of Profile for WEC
        prof: list of pref_voting Profile objects (needed for some axioms)
        axiom: string specifying the axiom ("default", "anonymity", "neutrality", "condorcet", "pareto", "independence")
        lambda_axiom: weight for the axiom loss
    """
    if axiom == "default":
        return 0
    elif axiom == "anonymity":
        axiom_loss = anonymity_loss(model, batch_X, num_samples=5)
        return lambda_axiom * axiom_loss
    elif axiom == "neutrality":
        axiom_loss = neutrality_loss(model, batch_X, model.max_cand, model.max_vot, num_samples=5)
        return lambda_axiom * axiom_loss
    elif axiom == "condorcet":
        axiom_loss = condorcet_loss(model, batch_X, prof, model.max_cand, model.max_vot)
        return lambda_axiom * axiom_loss
    elif axiom == "pareto":
        axiom_loss = pareto_loss(model, batch_X, prof, model.max_cand, model.max_vot)
        return lambda_axiom * axiom_loss
    elif axiom == "independence":
        axiom_loss = independence_loss(model, batch_X, prof, model.max_cand, model.max_vot, num_samples=5)
        return lambda_axiom * axiom_loss
    else:
        raise ValueError(f"Axiom {axiom} not supported.")
