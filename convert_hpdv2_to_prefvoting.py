import json
from pref_voting.profiles import Profile


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
            rankings.append(ranked_alternatives) # List of alternatives in order of their preference: [2, 0, 1] means, alt 2 is preferred over alt 0, which is preferred over alt 1

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


if __name__ == "__main__":
    # Laden und konvertieren der Daten
    file_path = r"99218df700704ef3e75bfc1510d5f4a97f74a5737d944674aca84c7057d920b7"

    # Konvertierung zu pref_voting Profilen
    profiles = load_and_convert(file_path)

    # Anzeige der Ergebnisse
    for i, profile in enumerate(profiles):
        print(f"Profile {i + 1}:")
        print(f"  Prompt: {profile.prompt}")
        print(f"  Num Alternatives: {profile.num_cands}")
        print(f"  Num Voters: {profile.num_voters}")
        print(f"  First Preference: {profile.rankings[0]}")
        print("-" * 50)