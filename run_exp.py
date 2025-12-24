#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Comparative Experiment Script for ReChorus
Comparing SelfGNN with baseline models (SASRec, LightGCN) on multiple datasets.
"""

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# ========== Configuration ==========
REPO_ROOT = Path(__file__).parent.absolute()

# Common hyperparameters
COMMON_ARGS = {
    "epoch": 1,              # Quick test: 1 epoch only
    "batch_size": 2048,      # 8GB VRAM: can use larger batch for faster training
    "num_workers": 0,        # Must be 0 on Windows
    "gpu": 0,
    "pin_memory": 1,
}

# Models to compare
MODELS = ["SelfGNN", "SASRec", "LightGCN"]

# Datasets to evaluate on (with processed data files)
DATASETS = [
    "Grocery_and_Gourmet_Food",
    "MovieLens_1M",
]

# Model-specific arguments
MODEL_ARGS = {
    "SelfGNN": {
        "time_periods": 6,
        "gnn_layer": 2,
        "att_layer": 2,
        "emb_size": 64,
        "num_layers": 1,
        "num_heads": 4,
        "ssl_weight": 1e-6,
    },
    "SASRec": {
        "emb_size": 64,
        "num_layers": 2,
        "num_heads": 4,
    },
    "LightGCN": {
        "emb_size": 64,
        "num_layers": 3,
    },
}


def build_command(model_name: str, dataset: str) -> list:
    """
    Build the training command for a specific model and dataset.
    
    Args:
        model_name: Name of the model (SelfGNN, SASRec, LightGCN)
        dataset: Name of the dataset
    
    Returns:
        Command as list of strings (safer for subprocess on Windows)
    """
    cmd_parts = [
        sys.executable,  # Use current Python interpreter
        str(REPO_ROOT / "src" / "main.py"),
        "--model_name", model_name,
        "--dataset", dataset,
    ]
    
    # Add common arguments
    for key, value in COMMON_ARGS.items():
        cmd_parts.extend([f"--{key}", str(value)])
    
    # Add model-specific arguments
    if model_name in MODEL_ARGS:
        for key, value in MODEL_ARGS[model_name].items():
            cmd_parts.extend([f"--{key}", str(value)])
    
    return cmd_parts


def run_experiment(model_name: str, dataset: str, cmd: list) -> bool:
    """
    Run a single experiment.
    Logs and models are saved by main.py to ../log/ and ../model/ directories.
    
    Args:
        model_name: Name of the model
        dataset: Name of the dataset
        cmd: Command to execute (as list)
    
    Returns:
        True if successful, False otherwise
    """
    print(f"\n{'='*80}")
    print(f"Running: {model_name} on {dataset}")
    print(f"{'='*80}\n")
    
    try:
        process = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
        )
        process.wait()
        return process.returncode == 0
            
    except Exception as e:
        print(f"Error running experiment: {e}")
        return False


def main():
    """Main function to run all experiments."""
    print(f"\n{'#'*80}")
    print(f"# ReChorus Comparative Experiment")
    print(f"# Models: {', '.join(MODELS)}")
    print(f"# Datasets: {', '.join(DATASETS)}")
    print(f"# Common Args: {COMMON_ARGS}")
    print(f"{'#'*80}\n")
    
    results = {}
    total_experiments = len(MODELS) * len(DATASETS)
    completed = 0
    successful = 0
    
    # Run experiments for each dataset
    for dataset in DATASETS:
        results[dataset] = {}
        
        # Run each model on this dataset
        for model in MODELS:
            completed += 1
            print(f"\n[{completed}/{total_experiments}] Experiment in progress...")
            
            cmd = build_command(model, dataset)
            success = run_experiment(model, dataset, cmd)
            
            results[dataset][model] = "✓ PASS" if success else "✗ FAIL"
            if success:
                successful += 1
    
    # Print final summary
    print(f"\n\n{'#'*80}")
    print(f"# Experiment Summary")
    print(f"{'#'*80}")
    print(f"Total: {total_experiments} | Successful: {successful} | Failed: {total_experiments - successful}")
    print(f"\nResults by Dataset:\n")
    
    for dataset in DATASETS:
        print(f"  {dataset}:")
        for model in MODELS:
            status = results.get(dataset, {}).get(model, "N/A")
            print(f"    {model:12} {status}")
    
    print(f"\n{'#'*80}")
    print(f"Logs saved to: ../log/")
    print(f"Models saved to: ../model/")
    print(f"{'#'*80}\n")
    
    return 0 if successful == total_experiments else 1


if __name__ == "__main__":
    sys.exit(main())
