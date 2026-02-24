
import os
import sys
import json
import matplotlib.pyplot as plt
import glob
from pathlib import Path

def get_latest_results_file(results_dir):
    files = glob.glob(os.path.join(results_dir, "benchmark_comprehensive_*.json"))
    if not files:
        return None
    return max(files, key=os.path.getmtime)

def main():
    # 1. Find Data
    project_root = Path(__file__).parent.parent.parent
    results_dir = project_root / "results"
    
    result_file = get_latest_results_file(results_dir)
    if not result_file:
        print("No benchmark results found.")
        return

    print(f"Loading results from: {result_file}")
    with open(result_file, 'r') as f:
        data = json.load(f)

    results = data.get("results", [])
    if not results:
        print("No results entries found.")
        return

    # 2. Extract Data
    # We want to plot: Instance Index vs Metric
    # Metric Priority: Approximation Ratio > Best Satisfaction % > Expected Satisfaction %
    
    # Check what keys are available in the first result
    first_keys = results[0].keys()
    metric_key = "expected_satisfaction_percent"
    ylabel = "Expected Satisfaction %"
    
    if "approximation_ratio" in first_keys and results[0]["approximation_ratio"] is not None:
        metric_key = "approximation_ratio"
        ylabel = "Approximation Ratio"
    elif "best_satisfaction_percent" in first_keys:
        metric_key = "best_satisfaction_percent"
        ylabel = "Best Satisfaction %"

    print(f"Plotting Metric: {metric_key}")

    # Organize data by method
    # methods = {"greedy": {"x": [], "y": []}, "kamis": ...}
    methods = {}
    
    for entry in results:
        method = entry.get("config_label", entry.get("method", "unknown"))
        if method not in methods:
            methods[method] = {"x": [], "y": []}
            
        instance_idx = entry.get("instance_idx", 0)
        val = entry.get(metric_key, 0.0)
        
        # Handle NaN/None
        if val is None: val = 0.0
        # If val is string "NaN", Float it
        try:
            val = float(val)
        except:
            val = 0.0

        methods[method]["x"].append(instance_idx)
        methods[method]["y"].append(val)

    # 3. Plot
    plt.figure(figsize=(10, 6))
    
    markers = ['o', 'x', 's', '^', 'D']
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for i, (method, data) in enumerate(methods.items()):
        marker = markers[i % len(markers)]
        color = colors[i % len(colors)]
        plt.scatter(data["x"], data["y"], label=method, alpha=0.7, edgecolors='none' if marker=='o' else None, marker=marker, c=color)

    plt.xlabel("Instance Index")
    plt.ylabel(ylabel)
    plt.title(f"Benchmark Results: {ylabel} vs Instance")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Save
    timestamp = Path(result_file).stem.replace("benchmark_comprehensive_", "")
    output_path = results_dir / f"scatter_plot_{timestamp}.png"
    plt.savefig(output_path)
    print(f"Plot saved to: {output_path}")

if __name__ == "__main__":
    main()
