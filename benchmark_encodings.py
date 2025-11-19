"""Benchmark script to compare pairwise, per-voter pairwise, and one-hot encodings for MLP models."""

import time
import json
import torch
from torch.utils.data import DataLoader

from principle_vote.synth_data import SynthData
from principle_vote.voting_mlp_pairwise import VotingMLP
from principle_vote.voting_mlp_pairwise_per_voter import VotingMLPPerVoter

torch.manual_seed(42)

# GPU Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    torch.backends.cudnn.benchmark = True  # Optimize for fixed input sizes
    # Set memory allocation strategy for better performance
    torch.cuda.empty_cache()
else:
    print("WARNING: CUDA not available. Running on CPU will be much slower!")

# Configuration
dataset_sizes = [1000, 5000, 15000, 50000, 150000, 500000, 1000000]  # Varying training set sizes
num_samples_test = 5000
max_num_candidates = 5
max_num_voters = 55
batch_size = 200
num_epochs = 3
prob_model = "IC"
winner_methods = ["borda", "plurality", "copeland"]  # Test multiple voting methods
encoding_types = ["pairwise", "pairwise_per_voter", "onehot"]  # Pairwise = symmetric encoding

results = {
    "config": {
        "dataset_sizes": dataset_sizes,
        "num_samples_test": num_samples_test,
        "max_candidates": max_num_candidates,
        "max_voters": max_num_voters,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "prob_model": prob_model,
        "winner_methods": winner_methods,
        "encoding_types": encoding_types
    }
}

def benchmark_encoding(encoding_type: str, winner_method: str, num_samples_train: int):
    """Benchmark a specific encoding type with a given winner method and dataset size."""
    print(f"\n{'='*60}")
    print(f"Benchmarking {encoding_type.upper()} encoding with {winner_method.upper()}")
    print(f"Dataset size: {num_samples_train:,} samples")
    print(f"{'='*60}\n")
    
    # Create training data
    print(f"Creating training data with {encoding_type} encoding...")
    start_time = time.time()
    data_train = SynthData(
        cand_max=max_num_candidates,
        vot_max=max_num_voters,
        num_samples=num_samples_train,
        prob_model=prob_model,
        winner_method=winner_method,
        encoding_type=encoding_type
    )
    mlp_dataset_train = data_train.encode_mlp()
    data_creation_time = time.time() - start_time
    print(f"Data creation time: {data_creation_time:.2f}s")
    
    # Create test data
    print(f"Creating test data with {encoding_type} encoding...")
    data_test = SynthData(
        cand_max=max_num_candidates,
        vot_max=max_num_voters,
        num_samples=num_samples_test,
        prob_model=prob_model,
        winner_method=winner_method,
        encoding_type=encoding_type
    )
    data_test.encode_mlp()
    mlp_X_test, mlp_y_test = data_test.get_encoded_mlp()
    
    # Get input size
    input_size = mlp_X_test.shape[1]
    print(f"Input size: {input_size}")
    
    # Create model
    print(f"Creating MLP model with {encoding_type} encoding...")
    
    train_loader = DataLoader(
        mlp_dataset_train,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=data_train.collate_profile,
        num_workers=0,  # Keep 0 for GPU; increase for CPU
        pin_memory=True if torch.cuda.is_available() else False,  # Faster data transfer to GPU
        persistent_workers=False
    )
    if encoding_type == "pairwise_per_voter":
        mlp_model = VotingMLPPerVoter(
            train_loader=train_loader,
            max_candidates=max_num_candidates,
            max_voters=max_num_voters,
        ).to(device)  # Move model to GPU
    else:
        mlp_model = VotingMLP(
            train_loader=train_loader,
            max_candidates=max_num_candidates,
            max_voters=max_num_voters,
            encoding_type=encoding_type
        ).to(device)  # Move model to GPU
    
    print(f"Model moved to {device}")
    
    # Count parameters
    num_params = sum(p.numel() for p in mlp_model.parameters())
    print(f"Number of parameters: {num_params:,}")
    
    # Calculate training steps based on epochs
    steps_per_epoch = (num_samples_train + batch_size - 1) // batch_size  # Ceiling division
    num_training_steps = num_epochs * steps_per_epoch
    print(f"Training for {num_epochs} epochs = {num_training_steps:,} steps ({steps_per_epoch} steps/epoch)")
    
    # Training
    start_time = time.time()
    mlp_model.train_model(num_steps=num_training_steps, seed=42, plot=False, axiom="none")
    training_time = time.time() - start_time
    print(f"Training time: {training_time:.2f}s ({training_time/num_training_steps*1000:.2f}ms per step)")
    
    # Evaluation
    print("Evaluating model...")
    start_time = time.time()
    # Move test data to GPU
    mlp_X_test_gpu = mlp_X_test.to(device)
    mlp_y_test_gpu = mlp_y_test.to(device)
    hard_accuracy = mlp_model.evaluate_model_hard(mlp_X_test_gpu, mlp_y_test_gpu)
    soft_accuracy = mlp_model.evaluate_model_soft(mlp_X_test_gpu, mlp_y_test_gpu)
    evaluation_time = time.time() - start_time
    print(f"Evaluation time: {evaluation_time:.2f}s")
    
    # Axiom satisfaction
    print("Checking axiom satisfaction...")
    axiom_types = ["anonymity", "neutrality", "condorcet", "pareto"]
    axiom_results = {}
    for axiom in axiom_types:
        try:
            satisfaction = mlp_model.evaluate_axiom_satisfaction(data_test, axiom=axiom)
            axiom_results[axiom] = satisfaction
        except Exception as e:
            print(f"Error checking {axiom}: {e}")
            axiom_results[axiom] = None
    
    # Inference time
    print("Measuring inference time...")
    mlp_model.eval()
    num_inference_runs = 100
    test_sample = mlp_X_test[0:1].to(device)
    
    # Warm-up run for GPU
    if torch.cuda.is_available():
        with torch.no_grad():
            for _ in range(10):
                _ = mlp_model.predict(test_sample)
        torch.cuda.synchronize()  # Ensure all GPU operations complete
    
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_inference_runs):
            _ = mlp_model.predict(test_sample)
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # Wait for all GPU operations to complete
    inference_time = (time.time() - start_time) / num_inference_runs * 1000
    print(f"Average inference time (single sample): {inference_time:.4f}ms")
    
    # Store results
    results[encoding_type] = {
        "input_size": input_size,
        "num_parameters": num_params,
        "data_creation_time": data_creation_time,
        "num_training_steps": num_training_steps,
        "num_epochs": num_epochs,
        "training_time": training_time,
        "training_time_per_step": training_time / num_training_steps,
        "training_time_per_epoch": training_time / num_epochs,
        "evaluation_time": evaluation_time,
        "inference_time_ms": inference_time,
        "hard_accuracy": hard_accuracy,
        "soft_accuracy": soft_accuracy,
        "axiom_satisfaction": axiom_results
    }
    
    print(f"\n{encoding_type.upper()} Results ({winner_method.upper()}):")
    print(f"  Hard Accuracy: {hard_accuracy:.4f}")
    print(f"  Soft Accuracy: {soft_accuracy:.4f}")
    print(f"  Training Time: {training_time:.2f}s")
    print(f"  Inference Time: {inference_time:.4f}ms")
    
    return mlp_model

# Benchmark all encodings across all winner methods and dataset sizes
print("Starting comprehensive benchmark comparison\n")
print(f"Testing with {len(winner_methods)} voting methods: {', '.join(winner_methods)}")
print(f"Testing with {len(encoding_types)} encoding types: {', '.join(encoding_types)}")
print(f"Testing with {len(dataset_sizes)} dataset sizes: {', '.join(f'{s:,}' for s in dataset_sizes)}\n")

for winner_method in winner_methods:
    print(f"\n{'#'*80}")
    print(f"# Testing with {winner_method.upper()} voting method")
    print(f"{'#'*80}")
    
    # Initialize results dictionary for this winner method
    results[winner_method] = {}
    
    for dataset_size in dataset_sizes:
        print(f"\n{'-'*80}")
        print(f"Dataset size: {dataset_size:,} samples")
        print(f"{'-'*80}")
        
        # Initialize results for this dataset size
        results[winner_method][str(dataset_size)] = {}
        
        for encoding_type in encoding_types:
            model = benchmark_encoding(encoding_type, winner_method, dataset_size)
            results[winner_method][str(dataset_size)][encoding_type] = results.pop(encoding_type)
            
            # Clear GPU memory after each model
            if torch.cuda.is_available():
                del model
                torch.cuda.empty_cache()
                print(f"GPU memory cleared. Allocated: {torch.cuda.memory_allocated()/1024**2:.2f}MB / {torch.cuda.get_device_properties(0).total_memory/1024**2:.2f}MB")

# Summary comparison for all voting methods
print(f"\n{'='*100}")
print("COMPREHENSIVE BENCHMARK SUMMARY")
print(f"{'='*100}\n")

for winner_method in winner_methods:
    print(f"\n{'='*100}")
    print(f"RESULTS FOR {winner_method.upper()} VOTING METHOD")
    print(f"{'='*100}\n")

    for dataset_size in dataset_sizes:
        print(f"\n{'='*100}")
        print(f"Dataset Size: {dataset_size:,} samples")
        print(f"{'='*100}\n")
        
        method_results = results[winner_method][str(dataset_size)]

    def print_row(label: str, values: list[str]):
        print(f"{label:<35}" + "".join(values))

    header_cells = [f"{enc.replace('_', ' ').title():<20}" for enc in encoding_types]
    print_row("Metric", header_cells)
    print(f"{'-'*35}" + ''.join('-' * 20 for _ in encoding_types))

    metric_specs = [
        ("Input Size", "input_size", "int"),
        ("Number of Parameters", "num_parameters", "int_commas"),
        ("Hard Accuracy", "hard_accuracy", "float"),
        ("Soft Accuracy", "soft_accuracy", "float"),
        ("Training Time (s)", "training_time", "float_2"),
        ("Time per Step (ms)", "training_time_per_step", "float_ms"),
        ("Inference Time (ms)", "inference_time_ms", "float_4"),
    ]

    def format_cell(value, kind: str) -> str:
        if value is None:
            return f"{'n/a':<20}"
        if kind == "int":
            return f"{int(value):<20}"
        if kind == "int_commas":
            with_commas = format(int(value), ",")
            return f"{with_commas:<20}"
        if kind == "float":
            return f"{float(value):<20.4f}"
        if kind == "float_2":
            return f"{float(value):<20.2f}"
        if kind == "float_ms":
            return f"{float(value) * 1000:<20.2f}"
        if kind == "float_4":
            return f"{float(value):<20.4f}"
        return f"{value:<20}"

    for label, key, kind in metric_specs:
        vals = [format_cell(method_results[enc].get(key), kind) for enc in encoding_types]
        print_row(label, vals)

    print(f"\n{'Axiom Satisfaction Rates':<35}")
    print(f"{'-'*35}" + ''.join('-' * 20 for _ in encoding_types))
    for axiom in ["anonymity", "neutrality", "condorcet", "pareto"]:
        vals = [
            format_cell(method_results[enc]['axiom_satisfaction'].get(axiom), "float")
            if method_results[enc]['axiom_satisfaction'].get(axiom) is not None else f"{'n/a':<20}"
            for enc in encoding_types
        ]
        print_row(axiom.capitalize(), vals)

    baseline = method_results.get("onehot")
    if baseline:
        print(f"\n{'Improvement vs One-hot (%)':<35}")
        print(f"{'-'*35}" + ''.join('-' * 20 for _ in encoding_types))
        improvement_metrics = [
            ("Input Size", "input_size", False),
            ("Num Parameters", "num_parameters", False),
            ("Hard Accuracy", "hard_accuracy", True),
            ("Soft Accuracy", "soft_accuracy", True),
            ("Training Time", "training_time", False),
            ("Inference Time", "inference_time_ms", False),
        ]

        def pct_change(value, reference, higher_is_better):
            if reference in (0, None) or value is None:
                return None
            if higher_is_better:
                return (value - reference) / reference * 100
            return (1 - value / reference) * 100

        for label, key, higher in improvement_metrics:
            vals = []
            for enc in encoding_types:
                val = method_results[enc].get(key)
                delta = pct_change(val, baseline.get(key), higher)
                if delta is None or enc == "onehot":
                    vals.append(f"{'0.00':<20}")
                else:
                    vals.append(f"{delta:>+20.2f}")
            print_row(label, vals)

# Overall comparison tables across dataset sizes
print(f"\n\n{'='*100}")
print("OVERALL COMPARISON - HARD ACCURACY BY DATASET SIZE")
print(f"{'='*100}\n")

for winner_method in winner_methods:
    print(f"\n{winner_method.upper()} Voting Method:")
    header = f"{'Dataset Size':<20}" + "".join(f"{enc.replace('_', ' ').title():<20}" for enc in encoding_types)
    print(header)
    print(f"{'-'*100}")
    for dataset_size in dataset_sizes:
        row = f"{dataset_size:<20,}"
        for enc in encoding_types:
            row += f"{results[winner_method][str(dataset_size)][enc]['hard_accuracy']:<20.4f}"
        print(row)

print(f"\n\n{'='*100}")
print("OVERALL COMPARISON - SOFT ACCURACY BY DATASET SIZE")
print(f"{'='*100}\n")

for winner_method in winner_methods:
    print(f"\n{winner_method.upper()} Voting Method:")
    header = f"{'Dataset Size':<20}" + "".join(f"{enc.replace('_', ' ').title():<20}" for enc in encoding_types)
    print(header)
    print(f"{'-'*100}")
    for dataset_size in dataset_sizes:
        row = f"{dataset_size:<20,}"
        for enc in encoding_types:
            row += f"{results[winner_method][str(dataset_size)][enc]['soft_accuracy']:<20.4f}"
        print(row)

print(f"\n\n{'='*100}")
print("OVERALL COMPARISON - TRAINING TIME (s) BY DATASET SIZE")
print(f"{'='*100}\n")

for winner_method in winner_methods:
    print(f"\n{winner_method.upper()} Voting Method:")
    header = f"{'Dataset Size':<20}" + "".join(f"{enc.replace('_', ' ').title():<20}" for enc in encoding_types)
    print(header)
    print(f"{'-'*100}")
    for dataset_size in dataset_sizes:
        row = f"{dataset_size:<20,}"
        for enc in encoding_types:
            row += f"{results[winner_method][str(dataset_size)][enc]['training_time']:<20.2f}"
        print(row)

# Save results to JSON
output_file = "benchmark_results_full.json"
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n\nDetailed results saved to {output_file}")

# Also create a summary CSV for easy analysis
print("\nCreating summary CSV files...")
import csv

# Create CSV for hard accuracy
with open("benchmark_hard_accuracy.csv", 'w', newline='') as f:
    writer = csv.writer(f)
    header = ["Voting Method", "Dataset Size"] + encoding_types
    writer.writerow(header)
    for winner_method in winner_methods:
        for dataset_size in dataset_sizes:
            row = [winner_method, dataset_size]
            for enc in encoding_types:
                row.append(results[winner_method][str(dataset_size)][enc]['hard_accuracy'])
            writer.writerow(row)

# Create CSV for training time
with open("benchmark_training_time.csv", 'w', newline='') as f:
    writer = csv.writer(f)
    header = ["Voting Method", "Dataset Size"] + encoding_types
    writer.writerow(header)
    for winner_method in winner_methods:
        for dataset_size in dataset_sizes:
            row = [winner_method, dataset_size]
            for enc in encoding_types:
                row.append(results[winner_method][str(dataset_size)][enc]['training_time'])
            writer.writerow(row)

print("Summary CSV files created: benchmark_hard_accuracy.csv, benchmark_training_time.csv")
