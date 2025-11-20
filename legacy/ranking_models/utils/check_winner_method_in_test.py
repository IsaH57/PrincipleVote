"""This script compares the winners found by "rank" in the test.json with the winners found when applying different winner selection methods on human voting results."""

import json

# Load data
with open('../test_transformed.json', 'r', encoding='utf-8') as f:
    test_data = json.load(f)

with open('../ranking_results/human_voting_analysis.json', 'r', encoding='utf-8') as f:
    human_data = json.load(f)

# Create a lookup for test winners
test_winners = {entry['prompt']: entry['winner'] for entry in test_data}

# Compare results
methods = ['borda', 'copeland', 'plurality']
results = {method: {'total': 0, 'match': 0} for method in methods}
output = []

for entry in human_data:
    prompt = entry['prompt']
    if prompt in test_winners:
        test_winner = test_winners[prompt]
        compare = {}
        for method in methods:
            human_winners = entry['winners'][method]
            match = test_winner in human_winners
            results[method]['total'] += 1
            if match:
                results[method]['match'] += 1
            compare[method] = {
                'test_winner': test_winner,
                'human_winners': human_winners,
                'match': match
            }
        output.append({
            'prompt': prompt,
            'comparison': compare
        })

# Calculate percentages
percentages = {method: (results[method]['match'] / results[method]['total'] * 100 if results[method]['total'] > 0 else 0)
               for method in methods}

final_output = {
    'comparisons': output,
    'percentages': percentages
}

with open('../ranking_results/winner_method_comparison_in_test.json', 'w', encoding='utf-8') as f:
    json.dump(final_output, f, indent=2, ensure_ascii=False)