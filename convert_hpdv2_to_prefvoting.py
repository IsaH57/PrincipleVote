import json

from pref_voting.c1_methods import copeland
from pref_voting.dominance_axioms import condorcet_winner
from pref_voting.profiles import Profile
from pref_voting.scoring_methods import borda, plurality
import torch
from torch import Tensor

import axioms


def convert_to_pref_voting_profiles(data):
    """Convert data to pref_voting Profile objects.

    Args:
        data: List of dictionaries containing 'prompt', 'image_path', and 'raw_annotations'.

    Returns:
        List of pref_voting Profile objects with rankings and metadata.
    """
    profiles = []

    for item in data:
        # Extract annotations
        annotations = [ann["annotation"] for ann in item["raw_annotations"]]

        num_alternatives = len(item["image_path"])

        # Convert ranking to preference (lower number = higher preference)
        rankings = []
        for annotation in annotations:
            # Create ranking based on annotation scores
            ranked_alternatives = sorted(range(num_alternatives), key=lambda x: annotation[x])
            rankings.append(
                ranked_alternatives)  # List of alternatives in order of their preference: [2, 0, 1] means, alt 2 is preferred over alt 0, which is preferred over alt 1

        # Create pref_voting profile
        profile = Profile(rankings)

        # Add metadata
        profile.prompt = item["prompt"]
        profile.image_paths = item["image_path"]
        profile.alternatives_name = {i: path for i, path in enumerate(item["image_path"])}

        profiles.append(profile)

    return profiles


def load_and_convert(file_path):
    """Load data from file and convert to pref_voting Profile objects.

    Args:
        file_path (str): Path to the file containing the data.

    Returns:
        List of pref_voting Profile objects.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return convert_to_pref_voting_profiles(data)


def make_winner_array(profile: Profile, voting_method: str):
    """Create an array indicating the winner for each profile using the specified voting method.

    Args:
        profile (Profile): The pref_voting Profile object.
        voting_method (str): The voting method function to determine the winner.

    Returns:
        List[int]: Array with 1 at the index of the winning alternative, 0 elsewhere.
    """
    if voting_method == "borda":
        winner_index = borda(profile)
    elif voting_method == "copeland":
        winner_index = copeland(profile)
    elif voting_method == "plurality":
        winner_index = plurality(profile)
    else:
        raise ValueError("Unsupported voting method")

    winner_array = torch.zeros(profile.num_cands, dtype=torch.int)
    for idx in winner_index:
        if 0 <= idx < profile.num_cands:
            winner_array[idx] = 1
    return winner_array


def write_to_json(profiles: list[Profile]):
    """Write analysis results to a JSON file.

    Args:
        profiles (List[Profile]): List of pref_voting Profile objects.

    """

    results = []

    for i, prof in enumerate(profiles):
        borda_winner = borda(prof)
        copeland_winner = copeland(prof)
        plurality_winner = plurality(prof)

        anonymity = axioms.check_anonymity(prof, make_winner_array(prof, "borda"), cand_max=10)
        neutrality = axioms.check_neutrality(prof, make_winner_array(prof, "borda"), cand_max=10)
        condorcet = axioms.check_condorcet(prof, make_winner_array(prof, "borda"), cand_max=10)
        pareto = axioms.check_pareto(prof, make_winner_array(prof, "borda"), cand_max=10)
        independence = axioms.check_independence(prof, make_winner_array(prof, "borda"), cand_max=10)

        profile_result = {
            "profile_id": i,
            "prompt": prof.prompt,
            "num_candidates": int(prof.num_cands),
            "num_voters": int(prof.num_voters),
            "winners": {
                "borda": [int(x) for x in borda_winner],
                "copeland": [int(x) for x in copeland_winner],
                "plurality": [int(x) for x in plurality_winner]
            },
            "axioms": {
                "anonymity": int(anonymity),
                "neutrality": int(neutrality),
                "condorcet": int(condorcet),
                "pareto": int(pareto),
                "independence": int(independence)
            }
        }

        results.append(profile_result)

    output_file = "voting_analysis_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Results saved as {output_file}")


if __name__ == "__main__":
    file_path = r"test.json"

    profiles = load_and_convert(file_path)

    write_to_json(profiles)
    """
    for prof in profiles:
        borda_winner = borda(prof)
        print("Borda winner index:", borda_winner)
        copeland_winner = copeland(prof)
        print("Copeland winner index:", copeland_winner)
        plurality_winner = plurality(prof)
        print("Plurality winner index:", plurality_winner)

        anonymity = axioms.check_anonymity(prof, make_winner_array(prof, "borda"), cand_max=10)
        print("Anonymity satisfied:", anonymity)
        neutrality = axioms.check_neutrality(prof, make_winner_array(prof, "borda"), cand_max=10)
        print("Neutrality satisfied:", neutrality)
        condorcet = axioms.check_condorcet(prof, make_winner_array(prof, "borda"), cand_max=10)
        print("Condorcet winner satisfied:", condorcet)
        pareto = axioms.check_pareto(prof, make_winner_array(prof, "borda"), cand_max=10)
        print("Pareto efficiency satisfied:", pareto)
        independence = axioms.check_independence(prof, make_winner_array(prof, "borda"), cand_max=10)
        print("Independence of irrelevant alternatives satisfied:", independence)

        print(f"Profile {i + 1}:")
        print(f"  Prompt: {profile.prompt}")
        print(f"  Num Alternatives: {profile.num_cands}")
        print(f"  Num Voters: {profile.num_voters}")
        print(f"  First Preference: {profile.rankings[0]}")
        print("-" * 50)
    """
