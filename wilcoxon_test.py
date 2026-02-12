#!/usr/bin/env python3
"""
Detailed Analysis of Benchmark Results
Computes Wilcoxon signed-rank tests and Win/Loss/Tie counts for Greedy vs KaMIS.
Grouped by Instance Type (Balanced vs Triangle).
"""

import json
import argparse
import sys
import numpy as np
from scipy import stats

def load_data(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    return data

def extract_paired_data(data, metrics):
    # Mapping: type -> {metric -> {'greedy': [], 'kamis': []}}
    # type 'all' will be aggregate
    results = {
        'balanced': {m: {'g': [], 'k': []} for m in metrics},
        'triangle': {m: {'g': [], 'k': []} for m in metrics},
        'all':      {m: {'g': [], 'k': []} for m in metrics}
    }
    
    # Detect labels
    seen_labels = set()
    for entry in data.get("results", []):
         seen_labels.add(entry.get("config_label", ""))
    
    greedy_label = next((l for l in seen_labels if "Greedy" in l), None)
    kamis_label = next((l for l in seen_labels if "KaMIS" in l), None)
    
    if not greedy_label or not kamis_label:
        print(f"Warning: Could not identify both Greedy and KaMIS labels. Found: {seen_labels}")
        return results

    # helper to find pairs
    # instance_idx is unique within a type run (usually) but safer to key by (n_vars, type, idx, seed)
    # The benchmark script saves 'type' and 'instance_idx'
    
    # Store by key
    grouped = {}
    
    for entry in data.get("results", []):
        label = entry.get("config_label", "")
        if label not in (greedy_label, kamis_label):
            continue
            
        # Key: (n_vars, type, instance_idx, seed)
        # Seed ensures we are comparing exact same instance even if indices are reused
        key = (entry.get("n_vars"), entry.get("type"), entry.get("instance_idx"), entry.get("seed"))
        
        if key not in grouped:
            grouped[key] = {}
        grouped[key][label] = entry

    # populate lists
    for key, pairs in grouped.items():
        if greedy_label in pairs and kamis_label in pairs:
            type_label = key[1] # "balanced" or "triangle"
            if type_label not in results:
                type_label = "unknown" 
                if type_label not in results: results[type_label] = {m: {'g': [], 'k': []} for m in metrics}
            
            g_entry = pairs[greedy_label]
            k_entry = pairs[kamis_label]
            
            for m in metrics:
                # Handle metric name variations
                val_g = g_entry.get(m)
                val_k = k_entry.get(m)
                
                # Special handling for satisfaction which might be named differently
                if m == "satisfaction_percent" and val_g is None:
                     val_g = g_entry.get("percent_satisfied_clauses")
                     val_k = k_entry.get("percent_satisfied_clauses")

                if val_g is not None and val_k is not None:
                    # Append to specific type
                    results[type_label][m]['g'].append(val_g)
                    results[type_label][m]['k'].append(val_k)
                    # Append to 'all'
                    results['all'][m]['g'].append(val_g)
                    results['all'][m]['k'].append(val_k)
                    
    return results

def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def analyze_metric(name, g_vals, k_vals, higher_is_better=True):
    n = len(g_vals)
    if n == 0:
        print(f"{name:<25} | No data")
        return

    g_arr = np.array(g_vals)
    k_arr = np.array(k_vals)
    diff = k_arr - g_arr
    
    # Win/Loss/Tie
    # Tolerance for float comparison
    tol = 1e-9
    
    if higher_is_better:
        kamis_wins = np.sum(diff > tol)
        greedy_wins = np.sum(diff < -tol)
    else:
        # Lower is better (e.g. Layers?) - stick to higher is better for generic interpretation if metric is score-like
        # But here let's assume "Kamis Win" means Kamis > Greedy for now unless specified.
        # User asked "how many times kamis won", usually implies better metric.
        # For 'satisfaction' and 'gradient_sum', higher is better.
        # For 'layers', lower is better? The user context usually implies minimizing layers for same energy 
        # OR maximizing energy for same layers.
        # Let's stick to strict comparison: K > G, G > K.
        kamis_wins = np.sum(diff > tol)
        greedy_wins = np.sum(diff < -tol)

    ties = np.sum(np.abs(diff) <= tol)
    
    # Wilcoxon
    # Zero differences are handled by scipy (discarded usually, or z-split)
    # We generally check if there are any differences first
    if np.all(np.abs(diff) <= tol):
        p_str = "N/A (All Tied)"
    else:
        try:
            stat, p = stats.wilcoxon(g_arr, k_arr)
            p_str = f"{p:.4e}"
        except Exception as e:
            p_str = "Error"

    print(f"{name:<25} | N={n:<4} | K>G: {kamis_wins:<4} | G>K: {greedy_wins:<4} | Ties: {ties:<4} | p-val: {p_str}")

def main():
    parser = argparse.ArgumentParser(description="Detailed Analysis of Benchmark Results")
    parser.add_argument("json_file", nargs="?", default="results/benchmark_comprehensive_2026-02-08-11-37-34.json")
    args = parser.parse_args()

    print(f"Loading {args.json_file}...")
    try:
        data = load_data(args.json_file)
    except FileNotFoundError:
        print("File not found.")
        sys.exit(1)

    metrics = [
        "satisfaction_percent", 
        "first_layer_gradient_sum", 
        "layers", 
        "time"
    ]
    
    results = extract_paired_data(data, metrics)
    
    # Process types
    for type_label in ["all", "balanced", "triangle"]:
        if type_label not in results: continue
        
        print_section(f"Instance Type: {type_label.upper()}")
        print(f"{'Metric':<25} | {'Count':<6} | {'Kamis >':<9} | {'Greedy >':<9} | {'Ties':<6} | {'Wilcoxon p'}")
        print("-" * 85)
        
        type_res = results[type_label]
        
        analyze_metric("Satisfaction (%)", type_res['satisfaction_percent']['g'], type_res['satisfaction_percent']['k'])
        analyze_metric("1st Layer Grad Sum", type_res['first_layer_gradient_sum']['g'], type_res['first_layer_gradient_sum']['k'])
        analyze_metric("Layers (Count)", type_res['layers']['g'], type_res['layers']['k'])
        analyze_metric("Time (s)", type_res['time']['g'], type_res['time']['k'])

if __name__ == "__main__":
    main()
