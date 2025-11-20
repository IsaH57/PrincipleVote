"""This script compares human voting results with model scoring results"""
import json

# Load data
with open('../ranking_results/human_voting_analysis.json', 'r', encoding='utf-8') as f:
    human_data = json.load(f)

with open('../ranking_results/model_scoring_results.json', 'r', encoding='utf-8') as f:
    model_data = json.load(f)

# Create a lookup for model winners
model_winners = {entry['prompt']: entry['relative_rankings'][0] for entry in model_data}

# Compare results
methods = ['borda', 'copeland', 'plurality']
results = {m: {'total': 0, 'match': 0, 'comparisons': []} for m in methods}

for entry in human_data:
    prompt = entry['prompt']
    winners = entry['winners']
    if prompt in model_winners:
        model_winner = model_winners[prompt]
        for method in methods:
            method_winners = winners.get(method, [])
            is_match = model_winner in method_winners
            results[method]['total'] += 1
            if is_match:
                results[method]['match'] += 1
            results[method]['comparisons'].append({
                'prompt': prompt,
                'method': method,
                'human_winners': method_winners,
                'model_winner': model_winner,
                'match': is_match
            })

# Calculate percentages
final_output = {}
for method in methods:
    total = results[method]['total']
    match = results[method]['match']
    percentage = (match / total * 100) if total > 0 else 0
    final_output[method] = {
        'percentage': percentage,
        'comparisons': results[method]['comparisons']
    }

with open('../result_comparisons/human_vs_model_scoring.json', 'w', encoding='utf-8') as f:
    json.dump(final_output, f, indent=2, ensure_ascii=False)