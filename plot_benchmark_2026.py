import json
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

def plot_benchmark_plots(json_file):
    print(f"Loading {json_file}...")
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File {json_file} not found.")
        return

    comparisons = {}
    
    # Check available results
    if "results" not in data:
        print("Error: No 'results' key in JSON.")
        return

    print("Found methods in JSON:")
    seen_labels = set()
    for r in data['results']:
        seen_labels.add(r.get('config_label', 'UNKNOWN'))
    print(seen_labels)

    # Dynamically find the labels
    greedy_label = next((l for l in seen_labels if "Greedy" in l), None)
    kamis_label = next((l for l in seen_labels if "KaMIS" in l), None)

    if not greedy_label or not kamis_label:
        print(f"Error: Could not identify both Greedy and KaMIS labels. Found: {seen_labels}")
        return

    print(f"Comparing: {greedy_label} vs {kamis_label}")

    for r in data['results']:
        label = r.get('config_label', '')
        if label not in [greedy_label, kamis_label]:
            continue
            
        key = (r['n_vars'], r['type'], r['instance_idx'])
        
        if key not in comparisons:
            comparisons[key] = {}
        
        # Store dict of metrics
        comparisons[key][label] = {
            'satisfaction': r['satisfaction_percent'],
            'layers': r['layers'],
            'energy': r['energy'],
            'first_layer_gradient_sum': r.get('first_layer_gradient_sum', 0.0)
        }

    # Extract paired data
    greedy_sat = []
    kamis_sat = []
    greedy_layers = []
    kamis_layers = []
    
    for key, methods in comparisons.items():
        if greedy_label in methods and kamis_label in methods:
            greedy_sat.append(methods[greedy_label]['satisfaction'])
            kamis_sat.append(methods[kamis_label]['satisfaction'])
            
            # For layers, we only care if they are tied, but let's collect all aligned
            greedy_layers.append(methods[greedy_label]['layers'])
            kamis_layers.append(methods[kamis_label]['layers'])

    if not greedy_sat:
        print("No matching instance pairs found.")
        return

    colors = {'balanced': 'blue', 'triangle': 'red', 'unknown': 'gray'}
    
    def plot_scatter_with_types(ax, greedy_data, kamis_data, types_data, label_prefix, title):
        # Organize data by type
        data_by_type = {}
        for g, k, t in zip(greedy_data, kamis_data, types_data):
            t = t if t in colors else 'unknown'
            if t not in data_by_type:
                data_by_type[t] = {'g': [], 'k': []}
            data_by_type[t]['g'].append(g)
            data_by_type[t]['k'].append(k)
            
        # Plot each type
        for t, d in data_by_type.items():
            ax.scatter(d['g'], d['k'], alpha=0.7, edgecolors='k', 
                       label=f"{t} ({len(d['g'])})", color=colors.get(t, 'gray'))
            
        # Limits and Diagonal
        all_vals = greedy_data + kamis_data
        if not all_vals: return
        min_val, max_val = min(all_vals), max(all_vals)
        buffer = (max_val - min_val) * 0.05 if max_val != min_val else 0.01
        lims = [min_val - buffer, max_val + buffer]
        
        ax.plot(lims, lims, 'r--', alpha=0.5, label="y=x")
        ax.set_xlabel(f"Greedy: {label_prefix}")
        ax.set_ylabel(f"KaMIS: {label_prefix}")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.legend()

    # --- Plot 1: Satisfaction ---
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Extract data with types
    g_sat, k_sat, types = [], [], []
    for key, methods in comparisons.items():
        if greedy_label in methods and kamis_label in methods:
            g_sat.append(methods[greedy_label]['satisfaction'])
            k_sat.append(methods[kamis_label]['satisfaction'])
            types.append(key[1]) # key is (n_vars, type, idx)

    plot_scatter_with_types(ax, g_sat, k_sat, types, "Satisfied Fraction", "KaMIS vs Greedy: Satisfaction")
    
    # Stats
    greedy_wins = sum(g > k for g, k in zip(g_sat, k_sat))
    kamis_wins = sum(k > g for g, k in zip(g_sat, k_sat))
    ties = sum(abs(k - g) < 1e-6 for g, k in zip(g_sat, k_sat))
    stats_text = f"Total: {len(g_sat)}\nGreedy Wins: {greedy_wins}\nKaMIS Wins: {kamis_wins}\nTies: {ties}"
    plt.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    if not os.path.exists("plots"):
        os.makedirs("plots")
    plt.savefig("plots/satisfaction_scatter_2026.png", dpi=300)
    print("Saved plots/satisfaction_scatter_2026.png")
    plt.close()

    # --- Plot 2: Layers (Ties Only) ---
    tied_g, tied_k, tied_types = [], [], []
    for i in range(len(g_sat)):
        if abs(g_sat[i] - k_sat[i]) < 1e-6:
            # Re-extract layers using same order
            key_tuple = list(comparisons.keys())[i] # Need to match index, tricky with dict...
            # Better to reconstruct list from items to ensure alignment
            pass 
            
    # Re-loop to ensure alignment
    g_layers, k_layers, l_types = [], [], []
    for key, methods in comparisons.items():
        if greedy_label in methods and kamis_label in methods:
            g_val = methods[greedy_label]['satisfaction']
            k_val = methods[kamis_label]['satisfaction']
            if abs(g_val - k_val) < 1e-6:
                g_layers.append(methods[greedy_label]['layers'])
                k_layers.append(methods[kamis_label]['layers'])
                l_types.append(key[1])

    if g_layers:
        fig, ax = plt.subplots(figsize=(8, 8))
        # Add jitter
        jitter = 0.1
        g_jit = [x + np.random.uniform(-jitter, jitter) for x in g_layers]
        k_jit = [x + np.random.uniform(-jitter, jitter) for x in k_layers]
        
        plot_scatter_with_types(ax, g_jit, k_jit, l_types, "Layers (Tied Instances)", f"KaMIS vs Greedy: Layers for {len(g_layers)} Tied Instances")
        plt.savefig("plots/layers_scatter_ties_2026.png", dpi=300)
        print("Saved plots/layers_scatter_ties_2026.png")
        plt.close()

    # --- Plot 3: First Layer Gradient Sum ---
    g_grads, k_grads, g_types = [], [], []
    for key, methods in comparisons.items():
        if greedy_label in methods and kamis_label in methods:
             g = methods[greedy_label].get('first_layer_gradient_sum', 0.0)
             k = methods[kamis_label].get('first_layer_gradient_sum', 0.0)
             if g > 0 or k > 0: # Filter empty
                 g_grads.append(g)
                 k_grads.append(k)
                 g_types.append(key[1])

    if g_grads:
        fig, ax = plt.subplots(figsize=(8, 8))
        plot_scatter_with_types(ax, g_grads, k_grads, g_types, "First Layer Gradient Sum", "KaMIS vs Greedy: 1st Layer Gradient Sum")
        
        equal_grads = sum(abs(g - k) < 1e-6 for g, k in zip(g_grads, k_grads))
        stats_text = f"Total: {len(g_grads)}\nEqual: {equal_grads}"
        plt.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                 
        plt.savefig("plots/gradient_sum_scatter_2026.png", dpi=300)
        print("Saved plots/gradient_sum_scatter_2026.png")
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot benchmark results")
    parser.add_argument("json_file", nargs="?", default="results/benchmark_comprehensive_2026-02-08-11-37-34.json")
    args = parser.parse_args()
    
    plot_benchmark_plots(args.json_file)
