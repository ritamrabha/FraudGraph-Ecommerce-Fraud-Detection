"""
Execution script to train, benchmark, and save all models and evaluation bundles.
Generates checkpoints/benchmark_amazon.json and checkpoints/benchmark_yelpchi.json.
"""

import os
import sys
import argparse
import time

# Ensure workspace root is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.train import run_full_benchmark


def main():
    parser = argparse.ArgumentParser(description="FraudGraph Benchmark Runner")
    parser.add_argument("--dataset", type=str, default="Amazon", choices=["Amazon", "YelpChi", "both"],
                        help="Dataset to benchmark (Amazon, YelpChi, or both)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Output directory for checkpoints")
    args = parser.parse_args()

    print("==================================================================")
    print("  FraudGraph: GNN-based Fraudulent User & Fake Review Detection   ")
    print("==================================================================")

    datasets = ["Amazon"] if args.dataset == "Amazon" else ["YelpChi"] if args.dataset == "YelpChi" else ["Amazon", "YelpChi"]

    for d in datasets:
        t0 = time.time()
        print(f"\n>>> Running Benchmark for: {d}")
        bundle = run_full_benchmark(dataset_name=d, seed=args.seed, save_dir=args.save_dir)
        print(f">>> Completed {d} in {round(time.time() - t0, 1)} seconds.")

    print("\n[SUCCESS] All requested benchmarks completed and checkpoints saved.")


if __name__ == "__main__":
    main()
