"""This script converts model ranking results from rank values to explicit rankings for each prompt.
It e.g. converts: [6,3,0,5,7,8,1,4,2] to [2,6,8,1,7,3,0,4,5].
"""
import json

# Load input JSON
with open("../ranking_results/model_ranking_results_raw_output.json", "r") as f:
    data = json.load(f)

output = {}

for prompt, ranks in data.items():
    # Sort indices by ascending rank value
    sorted_indices = sorted(range(len(ranks)), key=lambda i: ranks[i])
    output[prompt] = sorted_indices  # Only keep transformed ranking

# Save result
with open("../ranking_results/model_ranking_results.json", "w") as f:
    json.dump(output, f, indent=2)

print("Transformed JSON saved to model_ranking_results.json")
