
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import argparse
import pandas as pd
import matplotlib.patches as mpatches
import matplotlib

# Set global styles for paper-ready legibility
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 16,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
    'figure.titlesize': 20
})

def parse_operator_category(op_str):
    """
    Maps operator strings to logical categories:
    Mixer, X, Y, XX, YY, XY/YX, XZ/ZX, YZ/ZY
    """
    op_str = op_str.replace(" ", "")
    
    # 1. Standard Mixer (Sum of Xs across all/many qubits)
    if "+" in op_str and "X" in op_str and not ("Y" in op_str or "Z" in op_str):
        return "Mixer"
    
    # 2. Basics (Single-qubit)
    if op_str.startswith("X") and len(op_str) <= 3 and not ("Y" in op_str or "Z" in op_str):
        return "X"
    if op_str.startswith("Y") and len(op_str) <= 3 and not ("X" in op_str or "Z" in op_str):
        return "Y"
    
    # 3. Symmetric Entanglers (Products like X1X2)
    if "X" in op_str and "Y" not in op_str and "Z" not in op_str:
        return "XX"
    if "Y" in op_str and "X" not in op_str and "Z" not in op_str:
        return "YY"
    
    # 4. Cross-Basis (Problem-Aware)
    if ("X" in op_str and "Y" in op_str) and "Z" not in op_str:
        return "XY/YX"
    if ("X" in op_str and "Z" in op_str) and "Y" not in op_str:
        return "XZ/ZX"
    if ("Y" in op_str and "Z" in op_str) and "X" not in op_str:
        return "YZ/ZY"
    
    return "Other"

def generate_plots(results_dir, plots_dir):
    os.makedirs(plots_dir, exist_ok=True)
    
    json_files = glob.glob(os.path.join(results_dir, "**/*.json"), recursive=True)
    data_list = []
    
    for f in json_files:
        try:
            with open(f, 'r') as j:
                d = json.load(j)
                
                # --- STRATIFICATION & RENAMING LOGIC ---
                # Map legacy and ad-hoc labels to final standardized nomenclature
                method = d.get('method', 'unknown')
                
                # Back-filling legacy labels
                if method == "vanilla": method = "adapt_qaoa"
                if method == "vanilla_qaoa": method = "adapt_qaoa"
                if method == "vanilla_vqe": method = "adapt_vqe"
                if method == "regular_qaoa": method = "standard_qaoa"
                if method == "tetris_greedy" or method == "tetris_qaoa_greedy": method = "tetris_adapt_qaoa_greedy"
                if method == "tetris_kamis" or method == "mosaic_adapt_qaoa": method = "tetris_adapt_qaoa_kamis"

                # Handle files with generic "greedy"/"kamis" method field
                if method in ["greedy", "kamis"]:
                    if "regular_qaoa" in f or "standard_qaoa" in f:
                        method = "standard_qaoa"
                    elif "/vqe/" in f or "_vqe_" in f:
                        method = f"tetris_vqe_{method}"
                    elif "/qaoa/" in f or "_qaoa_" in f:
                        method = f"tetris_adapt_qaoa_{method}"
                
                d['method'] = method
                data_list.append(d)
        except:
            continue
            
    if not data_list:
        print("No data found!")
        return

    # --- FILTERING & MAPPING ---
    METHODS_TO_KEEP = ["adapt_qaoa", "tetris_adapt_qaoa_greedy", "tetris_adapt_qaoa_kamis"]
    DISPLAY_MAP = {
        "adapt_qaoa": "ADAPT-QAOA",
        "tetris_adapt_qaoa_greedy": "TETRIS-QAOA",
        "tetris_adapt_qaoa_kamis": "MosaicADAPT-QAOA"
    }

    # Filter data_list
    data_list = [d for d in data_list if d['method'] in METHODS_TO_KEEP]

    # Map methods to display names for consistency
    for d in data_list:
        d['method'] = DISPLAY_MAP.get(d['method'], d['method'])

    # Explicitly set method order for legend consistency
    preferred_order = ["ADAPT-QAOA", "TETRIS-QAOA", "MosaicADAPT-QAOA"]
    methods = [m for m in preferred_order if m in set(d['method'] for d in data_list)]
    
    # Dynamically find max layer depth across all traces by looking at the actual parameter arrays
    max_p = max([len(d.get('gamma_values', [])) for d in data_list] + [20])

    # --- 1. SAMPLE PERFORMANCE STATISTICS ---
    print("\n=== Sampled Bitstring Statistics (Approximation Ratio) ===")
    performance_by_method = {m: [] for m in methods}

    for d in data_list:
        method = d['method']
        # Use MEAN sampled satisfaction instead of BEST
        avg_sat = d.get('sampled_expected_satisfaction', 0)
        
        # Ground truth recovery
        gt = d.get('gurobi_energy')
        if gt is None or gt == 0:
            gt = d.get('num_clauses', 0)
        
        if gt == 0:
            exp_sat = d.get('sampled_expected_satisfaction', 0)
            pct = d.get('percent_satisfied_clauses', 0)
            if pct > 0:
                gt = round(exp_sat / pct)
        
        if gt > 0:
            ratio = avg_sat / gt
            performance_by_method[method].append(ratio)

    summary_data = []
    for method in methods:
        ratios = performance_by_method[method]
        if not ratios: continue
        
        res = {
            "Method": method,
            "Mean": np.mean(ratios),
            "Median": np.median(ratios),
            "Min": np.min(ratios),
            "Max": np.max(ratios),
            "Std": np.std(ratios),
            "Count": len(ratios)
        }
        summary_data.append(res)

    df_stats = pd.DataFrame(summary_data)
    print(df_stats.to_string(index=False))
    df_stats.to_csv(os.path.join(plots_dir, "sampled_statistics_summary.csv"), index=False)

    # Box Plot for Sampled Performance
    plt.figure(figsize=(14, 6))
    plot_data = []
    plot_labels = []
    for m in methods:
        if performance_by_method[m]:
            plot_data.append(performance_by_method[m])
            plot_labels.append(m)
    
    if plot_data:
        plt.boxplot(plot_data, labels=plot_labels if hasattr(plt, 'tick_labels') else plot_labels)
        plt.title("Sampled Performance: Best Approximation Ratio per Method")
        plt.ylabel("Sampled Best Approximation Ratio")
        plt.xticks(rotation=45)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "sampled_performance_boxplot.png"), dpi=300)
    plt.close()

    # --- 1.5. CONVERGENCE SPEED (LAYERS & PARAMETERS) TO 99.9% AR ---
    print("\n=== Convergence Speed to 99.9% Approximation Ratio ===")
    layers_by_method = {m: [] for m in methods}
    params_by_method = {m: [] for m in methods}
    
    for d in data_list:
        method = d['method']
        trace = d.get('clause_satisfaction_percent_trace', [])
        indices = d.get('selected_indices', [])
        
        # Find the first layer where the AR >= 0.999
        target_layer = None
        for i, ar in enumerate(trace):
            if ar >= 0.999:
                target_layer = i
                break
                
        if target_layer is not None:
            layers_by_method[method].append(target_layer)
            # Cumulative parameters up to and including target_layer
            subset = indices[:target_layer+1]
            if subset and isinstance(subset[0], list):
                total_params = sum(len(layer_indices) for layer_indices in subset)
            else:
                total_params = len(subset)
            params_by_method[method].append(total_params)
            
    summary_conv = []
    for method in methods:
        layers = layers_by_method[method]
        params = params_by_method[method]
        if not layers: continue
        
        res = {
            "Method": method,
            "Mean Layers": np.mean(layers),
            "Median Layers": np.median(layers),
            "Mean Params": np.mean(params),
            "Median Params": np.median(params),
            "Min Params": np.min(params),
            "Max Params": np.max(params),
            "Std Params": np.std(params),
            "Count (Reached Goal)": len(layers)
        }
        summary_conv.append(res)
        
    if summary_conv:
        df_conv = pd.DataFrame(summary_conv)
        print(df_conv.to_string(index=False))
        df_conv.to_csv(os.path.join(plots_dir, "layers_to_999ar_summary.csv"), index=False)

        # Plot 1: Layers Boxplot
        plt.figure(figsize=(14, 6))
        plot_data_layers = [layers_by_method[m] for m in methods if layers_by_method[m]]
        plot_labels_layers = [m for m in methods if layers_by_method[m]]
        if plot_data_layers:
            plt.boxplot(plot_data_layers, labels=plot_labels_layers)
            # plt.title("Layers Required to Reach 99.9% Approximation Ratio")
            plt.ylabel("Number of Layers (p)")
            plt.xticks(rotation=45)
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "layers_to_999ar_boxplot.png"), dpi=300)
        plt.close()

        # Plot 2: Parameters Boxplot
        plt.figure(figsize=(14, 6))
        plot_data_params = [params_by_method[m] for m in methods if params_by_method[m]]
        plot_labels_params = [m for m in methods if params_by_method[m]]
        if plot_data_params:
            plt.boxplot(plot_data_params, labels=plot_labels_params)
            plt.title("Total Parameters (Operators) Required to Reach 99.9% Approximation Ratio")
            plt.ylabel("Number of Parameters")
            plt.xticks(rotation=45)
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "parameters_to_999ar_boxplot.png"), dpi=300)
        plt.close()
    else:
        print("No recorded instances reached 99.9% AR in the provided runs.")

    # --- 1.6. LOCAL MINIMA / STUCK INSTANCES ANALYSIS ---
    print("\n=== Local Minima (Stuck Instances) Analysis ===")
    stuck_ar_by_method = {m: [] for m in methods}
    stuck_layer_by_method = {m: [] for m in methods}
    stuck_count_by_method = {m: 0 for m in methods}
    total_count_by_method = {m: 0 for m in methods}
    
    stoppers = ["SlowStopper", "ScoreStopper", "NoGradientAboveThreshold"]
    
    for d in data_list:
        method = d['method']
        total_count_by_method[method] += 1
        
        trace = d.get('clause_satisfaction_percent_trace', [])
        if not trace: continue
        
        final_ar = trace[-1]
        
        best_sat = d.get('sampled_best_satisfaction', 0)
        gt = d.get('gurobi_energy')
        if gt is None or gt == 0:
            gt = d.get('num_clauses', 0)
        
        if gt == 0:
            exp_sat = d.get('sampled_expected_satisfaction', 0)
            pct = d.get('percent_satisfied_clauses', 0)
            if pct > 0:
                gt = round(exp_sat / pct)
        
        if best_sat < gt:
            stuck_count_by_method[method] += 1
            stuck_ar_by_method[method].append(final_ar)
            stuck_layer_by_method[method].append(len(trace)-1) # depth index

    summary_stuck = []
    for method in methods:
        total = total_count_by_method[method]
        stuck = stuck_count_by_method[method]
        if total == 0: continue
        
        stuck_pct = (stuck / total) * 100.0
        ars = stuck_ar_by_method[method]
        layers = stuck_layer_by_method[method]
        
        res = {
            "Method": method,
            "Total Runs": total,
            "Stuck Count": stuck,
            "Stuck %": round(stuck_pct, 1),
            "Mean Stuck AR": np.mean(ars) if ars else np.nan,
            "Mean Stuck Layer": np.mean(layers) if layers else np.nan
        }
        summary_stuck.append(res)
        
    df_stuck = pd.DataFrame(summary_stuck)
    print(df_stuck.to_string(index=False))
    df_stuck.to_csv(os.path.join(plots_dir, "stuck_instances_summary.csv"), index=False)

    # Box Plot for Stuck AR
    plt.figure(figsize=(14, 6))
    plot_data_stuck_ar = []
    plot_labels_stuck_ar = []
    for m in methods:
        if stuck_ar_by_method[m]:
            plot_data_stuck_ar.append(stuck_ar_by_method[m])
            plot_labels_stuck_ar.append(m)
            
    if plot_data_stuck_ar:
        plt.boxplot(plot_data_stuck_ar, labels=plot_labels_stuck_ar if hasattr(plt, 'tick_labels') else plot_labels_stuck_ar)
        plt.title("Approximation Ratio of Local Minima Traps")
        plt.ylabel("Final AR (when strictly < 0.99)")
        plt.xticks(rotation=45)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "stuck_instances_ar_boxplot.png"), dpi=300)
    plt.close()

    # Box Plot for Stuck Layer Depth
    plt.figure(figsize=(14, 6))
    plot_data_stuck_lyr = []
    plot_labels_stuck_lyr = []
    for m in methods:
        if stuck_layer_by_method[m]:
            plot_data_stuck_lyr.append(stuck_layer_by_method[m])
            plot_labels_stuck_lyr.append(m)
            
    if plot_data_stuck_lyr:
        plt.boxplot(plot_data_stuck_lyr, labels=plot_labels_stuck_lyr if hasattr(plt, 'tick_labels') else plot_labels_stuck_lyr)
        plt.title("Layer Depth at which Local Minima Formed")
        plt.ylabel("Layer Depth (p)")
        plt.xticks(rotation=45)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "stuck_instances_depth_boxplot.png"), dpi=300)
    plt.close()

    # Bar Plot for Stuck Instance Counts
    plt.figure(figsize=(14, 6))
    counts = [stuck_count_by_method[m] for m in methods]
    
    bars = plt.bar(methods, counts, color='lightcoral', edgecolor='black')
    plt.title("Number of Instances Stuck in Local Minima")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.1, int(yval), ha='center', va='bottom', fontweight='bold')
    
    # Ensure y-axis has at least some range if all are 0
    if not any(counts):
        plt.ylim(0, 1) # Set a small range so the 0-level bars are visible
    else:
        # Give some padding at the top
        plt.ylim(0, max(counts) * 1.15 if max(counts) > 0 else 1)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "stuck_instances_count_barplot.png"), dpi=300)
    plt.close()


    # --- 2. CONVERGENCE PLOT ---
    def plot_convergence(max_layers, filename, title_suffix=""):
        plt.figure(figsize=(10, 6))
        for method in methods:
            m_data = [d for d in data_list if d['method'] == method]
            ar_matrix = []
            
            for d in m_data:
                trace = d.get('clause_satisfaction_percent_trace', [])
                if not trace: continue
                # We pad and slice up to max_layers + 1 to include layer 0 and layers 1 through max_layers
                padded = trace + [trace[-1]] * max(0, (max_layers + 1) - len(trace))
                ar_matrix.append(padded[:max_layers + 1])
                
            if not ar_matrix: continue
            ar_matrix = np.array(ar_matrix)
            avg_ar = np.mean(ar_matrix, axis=0)
            std_ar = np.std(ar_matrix, axis=0)
            
            p_range = range(0, max_layers + 1)
            plt.plot(p_range, avg_ar, label=method, marker='o' if max_layers <= 40 else '')
            plt.fill_between(p_range, avg_ar - std_ar, avg_ar + std_ar, alpha=0.1)

        # plt.title(f"Approximation Ratio vs Layer Depth {title_suffix}")
        plt.xlabel("Layer Depth (p)")
        plt.ylabel("Avg Approximation Ratio")
        plt.ylim(0.865, 1.01)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        # plt.xlim(0, max_layers)
        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, filename), dpi=300)
        plt.close()

    plot_convergence(20, "convergence_plot_full.png", "(Depth=20)")
    plot_convergence(min(40, max_p), "convergence_plot_40.png", "(First 40 Layers)")

    # --- 3. TRAP PLOT ---
    trap_max_p = min(20, max_p)  # Limit trap plot depth logic for readability
    plt.figure(figsize=(10, 6))
    for method in methods:
        m_data = [d for d in data_list if d['method'] == method]
        grad_matrix = []
        
        for d in m_data:
            trace = d.get('max_pool_gradients', [])
            if not trace: continue
            # We pad and slice up to trap_max_p + 1 to include layer 0 and layers 1 through trap_max_p
            padded = trace + [0.0] * max(0, (trap_max_p + 1) - len(trace))
            grad_matrix.append(padded[:trap_max_p + 1])
            
        if not grad_matrix: continue
        grad_matrix = np.array(grad_matrix)
        avg_grad = np.mean(grad_matrix, axis=0)
        std_grad = np.std(grad_matrix, axis=0)
        
        p_range = np.array(range(0, trap_max_p + 1))
        # Use errorbar instead of simple plot
        # For log scale, we need to be careful with yerr - lower bound cannot be <= 0
        yerr_lower = np.clip(avg_grad - 1e-15, 0, std_grad) # Don't let subtraction go below ~0
        plt.errorbar(p_range, avg_grad, yerr=std_grad, label=method, 
                     marker='s', capsize=3, linestyle='-', alpha=0.8)

    # plt.title("Maximum Gradient Magnitude of All Operators in Pool vs Layer Depth")
    plt.xlabel("Layer Depth (p)")
    plt.ylabel("Max Gradient Magnitude")
    plt.yscale('log')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.3)
    # plt.xlim(0, trap_max_p)
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "trap_plot.png"), dpi=300)
    plt.close()

    # --- 4. PER-METHOD OPERATOR EVOLUTION ---
    op_max_p = min(20, max_p) # Limit bar charts to 25 constraints for readability
    categories = ["Mixer", "X", "Y", "XX", "YY", "XY/YX", "XZ/ZX", "YZ/ZY"]
    palette = {
        "Mixer": "#000000", "X": "#D3D3D3", "Y": "#808080", "XX": "#87CEEB", "YY": "#4682B4",
        "XY/YX": "#FF6347", "XZ/ZX": "#FF4500", "YZ/ZY": "#8B0000", "Other": "#555555"
    }

    n_methods = len(methods)
    cols = min(3, n_methods)
    rows = (n_methods + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 5 * rows), sharey=True)
    if n_methods == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx, method in enumerate(methods):
        ax = axes[idx]
        m_data = [d for d in data_list if d['method'] == method]
        if not m_data:
            ax.set_title(f"{method} (No Data)")
            ax.axis('off')
            continue
        
        layer_counts = []
        for p in range(op_max_p):
            counts = {cat: 0 for cat in categories}
            for d in m_data:
                op_strings = d.get('selected_operator_strings', [])
                if p < len(op_strings):
                    op_str = op_strings[p]
                    if " + " in op_str:
                        sub_ops = op_str.split(" + ")
                        for so in sub_ops:
                            cat = parse_operator_category(so)
                            if cat in counts: counts[cat] += 1
                    else:
                        cat = parse_operator_category(op_str)
                        if cat in counts: counts[cat] += 1
            layer_counts.append(counts)

        bottom = np.zeros(op_max_p)
        p_range = np.arange(1, op_max_p + 1)

        valid_cats = [cat for cat in categories if any(lc[cat] > 0 for lc in layer_counts)]
        
        for cat in valid_cats:
            vals = np.array([lc[cat] for lc in layer_counts])
            totals = np.array([sum(lc.values()) for lc in layer_counts])
            pcts = np.divide(vals, totals, out=np.zeros_like(vals, dtype=float), where=totals!=0) * 100
            
            bars = ax.bar(p_range, pcts, bottom=bottom, label=cat, color=palette[cat], edgecolor='white', width=0.8)
            
            for i, bar in enumerate(bars):
                if pcts[i] > 10:
                    ax.text(bar.get_x() + bar.get_width()/2, 
                             bar.get_y() + bar.get_height()/2, 
                             f"{int(vals[i])}", 
                             ha='center', va='center', fontsize=7, 
                             color='white' if palette[cat] not in ["#D3D3D3", "#87CEEB"] else 'black')
            
            bottom += pcts

        ax.set_title(f"{method}")
        ax.set_xlabel("Layer Depth (p)")
        if idx % cols == 0:
            ax.set_ylabel("Relative Frequency (%)")
        else:
            ax.set_ylabel("")
            ax.tick_params(axis='y', labelleft=False)
        # Set ticks at 1, and then intervals of 5
        ax.set_xticks([1] + list(range(5, op_max_p + 1, 5)))
        ax.set_xlim(0.5, op_max_p + 0.5)
        ax.set_ylim(0, 105)

    # Hide any unused subplots
    for i in range(n_methods, len(axes)):
        axes[i].axis('off')

    # Create a global, normalized legend
    legend_patches = [mpatches.Patch(color=palette[cat], label=cat) for cat in categories]
    fig.legend(handles=legend_patches, loc='lower center', bbox_to_anchor=(0.5, -0.06), ncol=len(categories), title="Operator Categories", fontsize=16, title_fontsize=18)

    plt.tight_layout(rect=[0, 0.12, 1, 1])
    # plt.xlim(0, max_p+1)
    plt.savefig(os.path.join(plots_dir, "operator_evolution_combined.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # --- 5. KAMIS VS GREEDY LAYER SCATTER PLOTS ---
    def plot_kamis_vs_greedy_scatter(layer_depth, filename):
        plt.figure(figsize=(8, 8))
        x_greedy = []
        y_kamis = []
        
        # We need to trace TETRIS-QAOA and MosaicADAPT-QAOA
        # Pair them by instance_id
        greedy_data = {d['instance_id']: d for d in data_list if d['method'] == 'TETRIS-QAOA'}
        kamis_data = {d['instance_id']: d for d in data_list if d['method'] == 'MosaicADAPT-QAOA'}
        
        common_instances = set(greedy_data.keys()).intersection(set(kamis_data.keys()))
        
        for inst in common_instances:
            g_trace = greedy_data[inst].get('clause_satisfaction_percent_trace', [])
            k_trace = kamis_data[inst].get('clause_satisfaction_percent_trace', [])
            
            # The 0th entry is the 0-layer initial state, so Layer P is at index P
            g_val = g_trace[layer_depth] if len(g_trace) > layer_depth else (g_trace[-1] if g_trace else None)
            k_val = k_trace[layer_depth] if len(k_trace) > layer_depth else (k_trace[-1] if k_trace else None)
            
            if g_val is not None and k_val is not None:
                x_greedy.append(g_val)
                y_kamis.append(k_val)
                
        if not x_greedy:
            print(f"Skipping L={layer_depth} scatter plot, no overlapping KaMIS/Greedy data found.")
            plt.close()
            return
            
        plt.scatter(x_greedy, y_kamis, alpha=0.6, edgecolors='w', s=80)
        
        # Calculate Wins and Ties
        greedy_wins = 0
        kamis_wins = 0
        ties = 0
        tie_threshold = 0.0001 # Defined by user as same "1%" string (e.g. 97.01 vs 97.019 are basically 97.01)
        
        for g_val, k_val in zip(x_greedy, y_kamis):
            if abs(g_val - k_val) <= tie_threshold:
                ties += 1
            elif g_val > k_val:
                greedy_wins += 1
            else:
                kamis_wins += 1
        
        # Diagonal reference line (y = x)
        min_val = min(min(x_greedy), min(y_kamis))
        max_val = max(max(x_greedy), max(y_kamis))
        
        # Add some padding to the reference line
        pad = (max_val - min_val) * 0.1
        if pad == 0: pad = 0.05
        
        legend_txt = f'y=x\nTETRIS-QAOA Wins: {greedy_wins}\nMosaicADAPT-QAOA Wins: {kamis_wins}\nTies: {ties}'
        plt.plot([min_val - pad, max_val + pad], [min_val - pad, max_val + pad], 'k--', alpha=0.5, label=legend_txt)
        
        # plt.title(f"Expected Approximation Ratio at Layer {layer_depth}", fontsize=18)
        plt.xlabel("TETRIS-QAOA AR", fontsize=16)
        plt.ylabel("MosaicADAPT-QAOA AR", fontsize=16)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(loc='lower right', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, filename), dpi=300)
        plt.close()

    plot_kamis_vs_greedy_scatter(1, "kamis_vs_greedy_scatter_L1.png")
    plot_kamis_vs_greedy_scatter(2, "kamis_vs_greedy_scatter_L2.png")
    plot_kamis_vs_greedy_scatter(3, "kamis_vs_greedy_scatter_L3.png")

    print("\nAll plots and statistics generated in", plots_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir")
    parser.add_argument("plots_dir")
    args = parser.parse_args()
    generate_plots(args.results_dir, args.plots_dir)
