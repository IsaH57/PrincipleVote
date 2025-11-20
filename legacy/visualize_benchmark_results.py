"""Visualize benchmark results showing performance vs dataset size."""

import json
import matplotlib.pyplot as plt
import numpy as np

# Load benchmark results
with open('benchmark_results_full.json', 'r') as f:
    results = json.load(f)

config = results['config']
dataset_sizes = config['dataset_sizes']
winner_methods = config['winner_methods']
encoding_types = config['encoding_types']

# Color and marker schemes for each encoding
colors = {
    'pairwise': '#2E86AB',
    'pairwise_per_voter': '#A23B72',
    'onehot': '#F18F01'
}
markers = {
    'pairwise': 'o',
    'pairwise_per_voter': 's',
    'onehot': '^'
}
labels = {
    'pairwise': 'Pairwise (Aggregated)',
    'pairwise_per_voter': 'Pairwise (Per-Voter)',
    'onehot': 'One-Hot'
}

# Create figure with subplots for each voting method
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Hard Accuracy vs Dataset Size', fontsize=16, fontweight='bold', y=1.02)

for idx, method in enumerate(winner_methods):
    ax = axes[idx]
    
    for enc in encoding_types:
        accuracies = []
        sizes = []
        
        for size in dataset_sizes:
            try:
                acc = results[method][str(size)][enc]['hard_accuracy']
                accuracies.append(acc)
                sizes.append(size)
            except KeyError:
                pass
        
        ax.plot(sizes, accuracies, 
                marker=markers[enc], 
                color=colors[enc], 
                label=labels[enc],
                linewidth=2, 
                markersize=8,
                alpha=0.8)
    
    ax.set_xlabel('Training Dataset Size', fontsize=11, fontweight='bold')
    ax.set_ylabel('Hard Accuracy', fontsize=11, fontweight='bold')
    ax.set_title(f'{method.capitalize()}', fontsize=13, fontweight='bold')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='lower right', framealpha=0.9)
    ax.set_ylim([0, 1.05])

plt.tight_layout()
plt.savefig('hard_accuracy_vs_dataset_size.png', dpi=300, bbox_inches='tight')
print("Saved: hard_accuracy_vs_dataset_size.png")

# Soft Accuracy
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Soft Accuracy vs Dataset Size', fontsize=16, fontweight='bold', y=1.02)

for idx, method in enumerate(winner_methods):
    ax = axes[idx]
    
    for enc in encoding_types:
        accuracies = []
        sizes = []
        
        for size in dataset_sizes:
            try:
                acc = results[method][str(size)][enc]['soft_accuracy']
                accuracies.append(acc)
                sizes.append(size)
            except KeyError:
                pass
        
        ax.plot(sizes, accuracies, 
                marker=markers[enc], 
                color=colors[enc], 
                label=labels[enc],
                linewidth=2, 
                markersize=8,
                alpha=0.8)
    
    ax.set_xlabel('Training Dataset Size', fontsize=11, fontweight='bold')
    ax.set_ylabel('Soft Accuracy', fontsize=11, fontweight='bold')
    ax.set_title(f'{method.capitalize()}', fontsize=13, fontweight='bold')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='lower right', framealpha=0.9)
    ax.set_ylim([0, 1.05])

plt.tight_layout()
plt.savefig('soft_accuracy_vs_dataset_size.png', dpi=300, bbox_inches='tight')
print("Saved: soft_accuracy_vs_dataset_size.png")

# Axiom Satisfaction - Create separate plots for each axiom
axioms = ['anonymity', 'neutrality', 'condorcet', 'pareto', 'independence']

for axiom in axioms:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'{axiom.capitalize()} Satisfaction vs Dataset Size', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    for idx, method in enumerate(winner_methods):
        ax = axes[idx]
        
        for enc in encoding_types:
            satisfactions = []
            sizes = []
            
            for size in dataset_sizes:
                try:
                    sat = results[method][str(size)][enc]['axiom_satisfaction'][axiom]
                    if sat is not None:
                        satisfactions.append(sat)
                        sizes.append(size)
                except KeyError:
                    pass
            
            if satisfactions:
                ax.plot(sizes, satisfactions, 
                        marker=markers[enc], 
                        color=colors[enc], 
                        label=labels[enc],
                        linewidth=2, 
                        markersize=8,
                        alpha=0.8)
        
        ax.set_xlabel('Training Dataset Size', fontsize=11, fontweight='bold')
        ax.set_ylabel(f'{axiom.capitalize()} Satisfaction', fontsize=11, fontweight='bold')
        ax.set_title(f'{method.capitalize()}', fontsize=13, fontweight='bold')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='lower right' if axiom != 'condorcet' else 'best', framealpha=0.9)
        ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig(f'{axiom}_satisfaction_vs_dataset_size.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {axiom}_satisfaction_vs_dataset_size.png")

# Combined comparison plot - all metrics for one method
for method in winner_methods:
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'{method.capitalize()} - All Metrics vs Dataset Size', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    metrics = [
        ('hard_accuracy', 'Hard Accuracy'),
        ('soft_accuracy', 'Soft Accuracy'),
        ('anonymity', 'Anonymity'),
        ('neutrality', 'Neutrality'),
        ('condorcet', 'Condorcet'),
        ('pareto', 'Pareto'),
        ('independence', 'Independence')
    ]
    
    for plot_idx, (metric_key, metric_name) in enumerate(metrics):
        ax = axes[plot_idx // 3, plot_idx % 3]
        
        for enc in encoding_types:
            values = []
            sizes = []
            
            for size in dataset_sizes:
                try:
                    if metric_key in ['hard_accuracy', 'soft_accuracy']:
                        val = results[method][str(size)][enc][metric_key]
                    else:
                        val = results[method][str(size)][enc]['axiom_satisfaction'][metric_key]
                    
                    if val is not None:
                        values.append(val)
                        sizes.append(size)
                except KeyError:
                    pass
            
            if values:
                ax.plot(sizes, values, 
                        marker=markers[enc], 
                        color=colors[enc], 
                        label=labels[enc],
                        linewidth=2, 
                        markersize=7,
                        alpha=0.8)
        
        ax.set_xlabel('Dataset Size', fontsize=10, fontweight='bold')
        ax.set_ylabel(metric_name, fontsize=10, fontweight='bold')
        ax.set_title(metric_name, fontsize=11, fontweight='bold')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best', framealpha=0.9, fontsize=8)
        ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig(f'{method}_all_metrics_vs_dataset_size.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {method}_all_metrics_vs_dataset_size.png")

# Summary comparison - overlay all voting methods for each encoding
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Hard Accuracy Comparison Across Voting Methods', 
             fontsize=16, fontweight='bold', y=1.02)

method_colors = {
    'borda': '#2E86AB',
    'plurality': '#F18F01',
    'copeland': '#A23B72'
}

for enc_idx, enc in enumerate(encoding_types):
    ax = axes[enc_idx]
    
    for method in winner_methods:
        accuracies = []
        sizes = []
        
        for size in dataset_sizes:
            try:
                acc = results[method][str(size)][enc]['hard_accuracy']
                accuracies.append(acc)
                sizes.append(size)
            except KeyError:
                pass
        
        ax.plot(sizes, accuracies, 
                marker='o', 
                color=method_colors[method], 
                label=method.capitalize(),
                linewidth=2, 
                markersize=8,
                alpha=0.8)
    
    ax.set_xlabel('Training Dataset Size', fontsize=11, fontweight='bold')
    ax.set_ylabel('Hard Accuracy', fontsize=11, fontweight='bold')
    ax.set_title(labels[enc], fontsize=13, fontweight='bold')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='lower right', framealpha=0.9)
    ax.set_ylim([0, 1.05])

plt.tight_layout()
plt.savefig('hard_accuracy_by_voting_method.png', dpi=300, bbox_inches='tight')
print("Saved: hard_accuracy_by_voting_method.png")

print("\n✓ All visualizations generated successfully!")
print(f"Total plots created: 11")
