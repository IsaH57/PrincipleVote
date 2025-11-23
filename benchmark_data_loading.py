"""
Benchmark different data loading methods:
1. CSV loading with pandas
2. HuggingFace datasets format
3. Memory-mapped formats
4. Optimized CSV loading (chunked, dtypes, etc.)

Measures: loading time, memory usage, iteration speed
"""

import os
import time
import psutil
import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Tuple
import gc
import shutil

# Try to import datasets library
try:
    from datasets import load_from_disk, Dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Warning: HuggingFace datasets library not available")


class MemoryMonitor:
    """Monitor memory usage during operations"""
    
    def __init__(self):
        self.process = psutil.Process()
        self.baseline = None
        
    def start(self):
        """Record baseline memory"""
        gc.collect()
        self.baseline = self.process.memory_info().rss / 1024 / 1024  # MB
        
    def current_usage(self) -> float:
        """Get current memory delta in MB"""
        current = self.process.memory_info().rss / 1024 / 1024
        return current - self.baseline if self.baseline else current


class DataLoadingBenchmark:
    """Benchmark different data loading strategies"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.results = {}
        
    def benchmark_csv_pandas_default(self, filename: str = "personas.csv") -> Dict:
        """Benchmark standard pandas CSV loading"""
        print(f"\n{'='*60}")
        print("Benchmarking: Pandas CSV (Default)")
        print(f"{'='*60}")
        
        filepath = os.path.join(self.data_dir, filename)
        mem_monitor = MemoryMonitor()
        
        # Loading time
        mem_monitor.start()
        start_time = time.time()
        df = pd.read_csv(filepath)
        load_time = time.time() - start_time
        load_memory = mem_monitor.current_usage()
        
        # Get file size
        file_size_mb = os.path.getsize(filepath) / 1024 / 1024
        
        # Iteration speed (sample 1000 rows)
        num_iterations = min(1000, len(df))
        start_time = time.time()
        for idx in range(num_iterations):
            _ = df.iloc[idx]
        iteration_time = (time.time() - start_time) / num_iterations * 1000  # ms per row
        
        results = {
            'method': 'Pandas CSV (Default)',
            'file_size_mb': file_size_mb,
            'load_time_sec': load_time,
            'load_memory_mb': load_memory,
            'num_rows': len(df),
            'num_cols': len(df.columns),
            'iteration_time_ms': iteration_time,
            'throughput_rows_per_sec': num_iterations / (iteration_time * num_iterations / 1000)
        }
        
        self._print_results(results)
        del df
        gc.collect()
        
        return results
    
    def benchmark_csv_pandas_optimized(self, filename: str = "personas.csv") -> Dict:
        """Benchmark optimized pandas CSV loading with dtypes and low_memory"""
        print(f"\n{'='*60}")
        print("Benchmarking: Pandas CSV (Optimized)")
        print(f"{'='*60}")
        
        filepath = os.path.join(self.data_dir, filename)
        mem_monitor = MemoryMonitor()
        
        # Loading time with optimizations
        mem_monitor.start()
        start_time = time.time()
        df = pd.read_csv(filepath, low_memory=False, engine='c')
        load_time = time.time() - start_time
        load_memory = mem_monitor.current_usage()
        
        file_size_mb = os.path.getsize(filepath) / 1024 / 1024
        
        # Iteration speed
        num_iterations = min(1000, len(df))
        start_time = time.time()
        for idx in range(num_iterations):
            _ = df.iloc[idx]
        iteration_time = (time.time() - start_time) / num_iterations * 1000
        
        results = {
            'method': 'Pandas CSV (Optimized)',
            'file_size_mb': file_size_mb,
            'load_time_sec': load_time,
            'load_memory_mb': load_memory,
            'num_rows': len(df),
            'num_cols': len(df.columns),
            'iteration_time_ms': iteration_time,
            'throughput_rows_per_sec': num_iterations / (iteration_time * num_iterations / 1000)
        }
        
        self._print_results(results)
        del df
        gc.collect()
        
        return results
    
    def benchmark_csv_chunked(self, filename: str = "personas.csv", chunksize: int = 10000) -> Dict:
        """Benchmark chunked CSV loading"""
        print(f"\n{'='*60}")
        print(f"Benchmarking: Pandas CSV (Chunked, size={chunksize})")
        print(f"{'='*60}")
        
        filepath = os.path.join(self.data_dir, filename)
        mem_monitor = MemoryMonitor()
        
        # Loading time
        mem_monitor.start()
        start_time = time.time()
        chunks = []
        for chunk in pd.read_csv(filepath, chunksize=chunksize):
            chunks.append(chunk)
        df = pd.concat(chunks, ignore_index=True)
        load_time = time.time() - start_time
        load_memory = mem_monitor.current_usage()
        
        file_size_mb = os.path.getsize(filepath) / 1024 / 1024
        
        # Iteration speed
        num_iterations = min(1000, len(df))
        start_time = time.time()
        for idx in range(num_iterations):
            _ = df.iloc[idx]
        iteration_time = (time.time() - start_time) / num_iterations * 1000
        
        results = {
            'method': f'Pandas CSV (Chunked {chunksize})',
            'file_size_mb': file_size_mb,
            'load_time_sec': load_time,
            'load_memory_mb': load_memory,
            'num_rows': len(df),
            'num_cols': len(df.columns),
            'iteration_time_ms': iteration_time,
            'throughput_rows_per_sec': num_iterations / (iteration_time * num_iterations / 1000)
        }
        
        self._print_results(results)
        del df, chunks
        gc.collect()
        
        return results
    
    def benchmark_huggingface_dataset(self, dataset_path: str) -> Dict:
        """Benchmark HuggingFace datasets format loading"""
        if not HF_AVAILABLE:
            print("\nSkipping HuggingFace benchmark - library not available")
            return None
            
        print(f"\n{'='*60}")
        print("Benchmarking: HuggingFace Dataset Format")
        print(f"{'='*60}")
        
        mem_monitor = MemoryMonitor()
        
        # Loading time
        mem_monitor.start()
        start_time = time.time()
        dataset = load_from_disk(dataset_path)
        
        # If it's a DatasetDict, get the train split
        if hasattr(dataset, 'keys'):
            dataset = dataset['train']
            
        load_time = time.time() - start_time
        load_memory = mem_monitor.current_usage()
        
        # Get dataset size (approximate)
        dataset_size_mb = sum(
            f.stat().st_size for f in Path(dataset_path).rglob('*') if f.is_file()
        ) / 1024 / 1024
        
        # Iteration speed
        num_iterations = min(1000, len(dataset))
        start_time = time.time()
        for idx in range(num_iterations):
            _ = dataset[idx]
        iteration_time = (time.time() - start_time) / num_iterations * 1000
        
        results = {
            'method': 'HuggingFace Dataset',
            'file_size_mb': dataset_size_mb,
            'load_time_sec': load_time,
            'load_memory_mb': load_memory,
            'num_rows': len(dataset),
            'num_cols': len(dataset.column_names),
            'iteration_time_ms': iteration_time,
            'throughput_rows_per_sec': num_iterations / (iteration_time * num_iterations / 1000)
        }
        
        self._print_results(results)
        del dataset
        gc.collect()
        
        return results
    
    def benchmark_csv_to_hf_conversion(self, csv_file: str) -> dict:
        """
        Convert a CSV file to a HuggingFace Dataset, benchmark the conversion
        and loading speed, then clean up the temporary cache directory.
        """
        if not HF_AVAILABLE:
            print("\nSkipping HuggingFace conversion benchmark - library not available")
            return None
            
        print(f"\n{'='*60}")
        print("Benchmarking: CSV to HuggingFace Conversion")
        print(f"{'='*60}")
        
        filepath = os.path.join(self.data_dir, csv_file)
        hf_cache_path = os.path.join(self.data_dir, "hf_cache_temp")
        
        # --------------------------------------------------------------
        # 1️⃣ Clean any previous temporary cache directory (robust)
        # --------------------------------------------------------------
        if os.path.isdir(hf_cache_path):
            try:
                # ignore_errors=True ensures removal even if the dir is not empty
                shutil.rmtree(hf_cache_path, ignore_errors=True)
                print(f"[INFO] Removed existing temporary cache directory: {hf_cache_path}")
            except Exception as e:
                print(f"[WARN] Failed to remove {hf_cache_path}: {e}")
        # --------------------------------------------------------------
        # 2️⃣ Load CSV with pandas (this part was already present)
        # --------------------------------------------------------------
        mem_monitor = MemoryMonitor()
        mem_monitor.start()
        
        start_load = time.time()
        df = pd.read_csv(filepath)
        load_time = time.time() - start_load
        # --------------------------------------------------------------
        # 3️⃣ Convert to HuggingFace Dataset and save to temporary cache
        # --------------------------------------------------------------
        start_conv = time.time()
        hf_dataset = Dataset.from_pandas(df)
        hf_dataset.save_to_disk(hf_cache_path)
        conversion_time = time.time() - start_conv
        # --------------------------------------------------------------
        # 4️⃣ Load the saved HuggingFace dataset (benchmark loading)
        # --------------------------------------------------------------
        start_hf_load = time.time()
        loaded_hf = load_from_disk(hf_cache_path)
        hf_load_time = time.time() - start_hf_load
        # --------------------------------------------------------------
        # 5️⃣ Gather results
        # --------------------------------------------------------------
        load_memory = mem_monitor.current_usage()
        file_size_mb = os.path.getsize(filepath) / 1024 / 1024
        
        # Iteration speed
        num_iterations = min(1000, len(loaded_hf))
        start_time = time.time()
        for idx in range(num_iterations):
            _ = loaded_hf[idx]
        iteration_time = (time.time() - start_time) / num_iterations * 1000
        
        result = {
            'method': 'CSV to HuggingFace',
            'file_size_mb': file_size_mb,
            'conversion_time_sec': conversion_time,
            'load_time_sec': hf_load_time,
            'load_memory_mb': load_memory,
            'num_rows': len(df),
            'num_cols': len(df.columns),
            'iteration_time_ms': iteration_time,
            'throughput_rows_per_sec': num_iterations / (iteration_time * num_iterations / 1000)
        }
        # --------------------------------------------------------------
        # 6️⃣ Final cleanup (again, robust)
        # --------------------------------------------------------------
        self._print_results(result)
        
        if os.path.isdir(hf_cache_path):
            shutil.rmtree(hf_cache_path, ignore_errors=True)
            
        del df, hf_dataset, loaded_hf
        gc.collect()
        
        return result
    
    def _print_results(self, results: Dict):
        """Pretty print benchmark results"""
        print(f"\nResults for: {results['method']}")
        print(f"  File/Dataset Size: {results.get('file_size_mb', 0):.2f} MB")
        if 'conversion_time_sec' in results:
            print(f"  Conversion Time: {results['conversion_time_sec']:.3f} sec")
        print(f"  Load Time: {results['load_time_sec']:.3f} sec")
        print(f"  Memory Usage: {results['load_memory_mb']:.2f} MB")
        print(f"  Rows: {results['num_rows']:,} | Columns: {results['num_cols']}")
        print(f"  Iteration Time: {results['iteration_time_ms']:.4f} ms/row")
        print(f"  Throughput: {results['throughput_rows_per_sec']:.2f} rows/sec")
    
    def run_all_benchmarks(self, csv_file: str = "personas.csv", hf_dataset_path: str = None):
        """Run all benchmarks and generate comparison report"""
        print("\n" + "="*60)
        print("DATA LOADING BENCHMARK SUITE")
        print("="*60)
        
        results = []
        
        # CSV benchmarks
        if os.path.exists(os.path.join(self.data_dir, csv_file)):
            results.append(self.benchmark_csv_pandas_default(csv_file))
            results.append(self.benchmark_csv_pandas_optimized(csv_file))
            results.append(self.benchmark_csv_chunked(csv_file, chunksize=10000))
            results.append(self.benchmark_csv_chunked(csv_file, chunksize=50000))
            
            # CSV to HF conversion
            if HF_AVAILABLE:
                results.append(self.benchmark_csv_to_hf_conversion(csv_file))
        
        # HuggingFace dataset benchmark
        if hf_dataset_path and os.path.exists(hf_dataset_path) and HF_AVAILABLE:
            results.append(self.benchmark_huggingface_dataset(hf_dataset_path))
        
        # Filter out None results
        results = [r for r in results if r is not None]
        
        # Generate comparison report
        self._generate_comparison_report(results)
        
        return results
    
    def _generate_comparison_report(self, results: List[Dict]):
        """Generate a comprehensive comparison report"""
        print("\n" + "="*60)
        print("BENCHMARK COMPARISON REPORT")
        print("="*60)
        
        if not results:
            print("No results to compare")
            return
        
        # Create comparison table
        print(f"\n{'Method':<30} {'Load Time':<12} {'Memory':<12} {'Iter Time':<15} {'Throughput':<15}")
        print("-" * 90)
        
        for r in results:
            method = r['method'][:28]
            load_time = f"{r['load_time_sec']:.3f}s"
            memory = f"{r['load_memory_mb']:.1f}MB"
            iter_time = f"{r['iteration_time_ms']:.4f}ms"
            throughput = f"{r['throughput_rows_per_sec']:.1f}r/s"
            
            print(f"{method:<30} {load_time:<12} {memory:<12} {iter_time:<15} {throughput:<15}")
        
        # Find best performers
        print("\n" + "="*60)
        print("BEST PERFORMERS")
        print("="*60)
        
        fastest_load = min(results, key=lambda x: x['load_time_sec'])
        print(f"Fastest Load: {fastest_load['method']} ({fastest_load['load_time_sec']:.3f}s)")
        
        lowest_memory = min(results, key=lambda x: x['load_memory_mb'])
        print(f"Lowest Memory: {lowest_memory['method']} ({lowest_memory['load_memory_mb']:.1f}MB)")
        
        fastest_iter = min(results, key=lambda x: x['iteration_time_ms'])
        print(f"Fastest Iteration: {fastest_iter['method']} ({fastest_iter['iteration_time_ms']:.4f}ms/row)")
        
        highest_throughput = max(results, key=lambda x: x['throughput_rows_per_sec'])
        print(f"Highest Throughput: {highest_throughput['method']} ({highest_throughput['throughput_rows_per_sec']:.1f}rows/s)")
        
        # Save results to JSON
        output_file = os.path.join(self.data_dir, "benchmark_results.json")
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_file}")


def main():
    """Main benchmark execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Benchmark data loading methods')
    parser.add_argument('--data-dir', default='data', help='Directory containing data files')
    parser.add_argument('--csv-file', default='personas.csv', help='CSV file to benchmark')
    parser.add_argument('--hf-dataset', default=None, help='Path to HuggingFace dataset directory')
    parser.add_argument('--method', choices=['csv', 'hf', 'all'], default='all', 
                       help='Which benchmark to run')
    
    args = parser.parse_args()
    
    benchmark = DataLoadingBenchmark(data_dir=args.data_dir)
    
    if args.method == 'all':
        benchmark.run_all_benchmarks(
            csv_file=args.csv_file,
            hf_dataset_path=args.hf_dataset
        )
    elif args.method == 'csv':
        benchmark.benchmark_csv_pandas_default(args.csv_file)
        benchmark.benchmark_csv_pandas_optimized(args.csv_file)
        benchmark.benchmark_csv_chunked(args.csv_file)
    elif args.method == 'hf' and args.hf_dataset:
        benchmark.benchmark_huggingface_dataset(args.hf_dataset)


if __name__ == "__main__":
    main()
