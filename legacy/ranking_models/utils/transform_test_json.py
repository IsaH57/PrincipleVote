"""This script transforms the test.json file by converting "rank" values into explicit rankings and identifying the winner image for each prompt.
It e.g. converts: [6,3,0,5,7,8,1,4,2] to [2,6,8,1,7,3,0,4,5].
"""
import json

# Load the input JSON
with open("../test.json", "r") as f:
    data = json.load(f)

output = []
for entry in data:
    ranks = entry["rank"]

    # Sort image indices by their rank values (ascending: lower rank = better)
    sorted_indices = sorted(range(len(ranks)), key=lambda i: ranks[i])

    # Winner = best image (lowest rank)
    winner = sorted_indices[0]

    # Create new entry
    transformed_entry = {
        "prompt": entry["prompt"],
        "ranking": sorted_indices,
        "winner": winner
    }

    output.append(transformed_entry)

# Save the transformed JSON
with open("../test_transformed.json", "w") as f:
    json.dump(output, f, indent=2)

print("Transformed JSON saved to test_transformed.json ")
