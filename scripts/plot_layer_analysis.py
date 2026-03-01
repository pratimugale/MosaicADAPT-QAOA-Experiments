import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
import json
import os
import glob
from matplotlib.backends.backend_pdf import PdfPages
import warnings
from scipy.stats import wilcoxon

# Import our custom composition analyzer
try:
    from scripts.analyze_operator_composition import get_operator_composition_data
except ImportError:
    from analyze_operator_composition import get_operator_composition_data

warnings.filterwarnings('ignore', category=UserWarning, module='seaborn')

def load_all_data(results_dir):
    """
    Loads data from all `_all_` JSON files in the given directory.
    Extracts the trace, calculates the approximation ratio, and computes forward-fills for L=1, 2, 3.
    """
    json_files = glob.glob(os.path.join(results_dir, "**", "*_all_*.json"), recursive=True)
    all_rows = []
    
    for file_path in json_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            for res in data.get('results', []):
                # Basic Properties
                method = res.get('method', 'unknown').lower()
                initial_gamma = res.get('initial_gamma', None)
                instance_idx = res.get('instance_idx', None)
                filename = res.get('filename', 'unknown')
                
                # Math Properties
                gtype = res.get('type', '').lower()
                if 'notriangle' in gtype or 'triangle' in gtype:
                    graph_type = 'Triangle'
                elif 'balanced' in gtype:
                    graph_type = 'Balanced'
                else:
                    is_triangle = "notrianglesat" in filename.lower()
                    is_balanced = "balancedsat" in filename.lower()
                    graph_type = "Triangle" if is_triangle else ("Balanced" if is_balanced else "Unknown")
                
                
                num_clauses = 0
                if filename and 'clauses' in filename:
                    try:
                        num_clauses = int(filename.split('clauses')[0].split('_')[-2])
                    except:
                        pass
                
                max_satisfied_gurobi = res.get('max_satisfied_gurobi', 0)
                if num_clauses == 0 or max_satisfied_gurobi == 0:
                    continue
                    
                gurobi_sat_percent = max_satisfied_gurobi / num_clauses
                if gurobi_sat_percent <= 0:
                    continue
                
                # Trace Parsing
                trace = res.get('adaptation_clause_satisfaction_percent_trace', [])
                
                # If trace is empty, we can't extract layer-wise data
                if not trace:
                    continue
                
                # Forward Fill Logic & Approx Ratio Calculation
                # trace[0] is L=0 (untrained)
                # trace[1] is L=1
                # trace[2] is L=2
                # trace[3] is L=3
                
                # Fill L1
                if len(trace) > 1:
                    sat_l1 = trace[1]
                else:
                    sat_l1 = trace[-1] # Forward fill the only available value (L0) if loop never ran
                approx_l1 = sat_l1 / gurobi_sat_percent
                
                # Fill L2
                if len(trace) > 2:
                    sat_l2 = trace[2]
                else:
                    sat_l2 = trace[-1] # Forward fill final achieved value
                approx_l2 = sat_l2 / gurobi_sat_percent
                
                # Fill L3
                if len(trace) > 3:
                    sat_l3 = trace[3]
                else:
                    sat_l3 = trace[-1] # Forward fill final achieved value
                approx_l3 = sat_l3 / gurobi_sat_percent
                
                # Extract Gradient Sum for Layer 1
                selected_scores = res.get('selected_scores', [])
                l1_gradient_sum = 0.0
                if selected_scores and len(selected_scores) > 0 and len(selected_scores[0]) > 0:
                    l1_gradient_sum = sum(selected_scores[0])
                
                row = {
                    'instance_idx': instance_idx,
                    'method': method,
                    'initial_gamma': initial_gamma,
                    'graph_type': graph_type,
                    'approx_l1': approx_l1,
                    'approx_l2': approx_l2,
                    'approx_l3': approx_l3,
                    'l1_gradient_sum': l1_gradient_sum
                }
                all_rows.append(row)
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            
    df = pd.DataFrame(all_rows)
    return df

def plot_layer_scatters(df, gamma, layer_depth, pdf):
    """
    Generates a single scatter plot for exactly one Gamma at exactly one Layer Depth comparing KaMIS vs Greedy.
    """
    # Filter by gamma
    df_gamma = df[df['initial_gamma'] == gamma]
    if df_gamma.empty:
        return
        
    # Isolate Greedy and KaMIS
    greedy_runs = df_gamma[df_gamma['method'] == 'greedy']
    kamis_runs = df_gamma[df_gamma['method'] == 'kamis']
    
    if greedy_runs.empty or kamis_runs.empty:
        return
        
    # We want max approx ratio in case multiple seeds/runs exist for the same pure instance under the same config
    # Though in `_all_` files, there should strictly be 1 greedy run and 1 kamis run per instance per gamma
    greedy_agg = greedy_runs.groupby(['instance_idx', 'graph_type'])[f'approx_l{layer_depth}'].max().reset_index()
    greedy_agg.rename(columns={f'approx_l{layer_depth}': 'greedy_val'}, inplace=True)
    
    kamis_agg = kamis_runs.groupby(['instance_idx', 'graph_type'])[f'approx_l{layer_depth}'].max().reset_index()
    kamis_agg.rename(columns={f'approx_l{layer_depth}': 'kamis_val'}, inplace=True)
    
    # Merge on instance entirely
    merged = pd.merge(greedy_agg, kamis_agg, on=['instance_idx', 'graph_type'])
    
    if merged.empty:
        return
        
    # Arrays
    g_val = merged['greedy_val']
    k_val = merged['kamis_val']
    
    # Calculate Wins (Tolerance of 1e-6 ratio = 0.0001%)
    tolerance = 1e-6
    kamis_wins = sum((k_val - g_val) > tolerance)
    greedy_wins = sum((g_val - k_val) > tolerance)
    ties = sum(abs(k_val - g_val) <= tolerance)
    
    legend_label = f'y=x (Tie)\nKaMIS Wins: {kamis_wins}\nGreedy Wins: {greedy_wins}\nTies (±0.0001%): {ties}'
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 8))
    
    balanced_merged = merged[merged['graph_type'] == 'Balanced']
    triangle_merged = merged[merged['graph_type'] == 'Triangle']
    
    if not balanced_merged.empty:
        ax.scatter(balanced_merged['greedy_val'], balanced_merged['kamis_val'], alpha=0.7, color='blue', marker='o', edgecolor='k', label='Balanced')
    if not triangle_merged.empty:
        ax.scatter(triangle_merged['greedy_val'], triangle_merged['kamis_val'], alpha=0.7, color='orange', marker='^', edgecolor='k', label='Triangle')
    
    min_val = min(g_val.min(), k_val.min()) - 0.01 if not g_val.empty else 0
    max_val = max(g_val.max(), k_val.max()) + 0.01 if not g_val.empty else 1
    if min_val == max_val: 
        min_val -= 0.1
        max_val += 0.1
        
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', label=legend_label)
    
    # Draw tolerance boundary
    import numpy as np
    x_vals = np.array([min_val, max_val])
    ax.fill_between(x_vals, x_vals - tolerance, x_vals + tolerance, color='red', alpha=0.1, label='Tie Region')
    
    ax.set_xlim([min_val, max_val])
    ax.set_ylim([min_val, max_val])
    ax.set_xlabel(f'Greedy Approx Ratio (L={layer_depth})')
    ax.set_ylabel(f'KaMIS Approx Ratio (L={layer_depth})')
    ax.set_title(f'L={layer_depth} Approx Ratio: KaMIS vs Greedy (Gamma = {gamma})')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='lower right')
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

def plot_l1_gradient_scatter(df, gamma, pdf):
    """
    Generates a scatter plot for the sum of gradients of operators chosen in the first layer.
    """
    df_gamma = df[df['initial_gamma'] == gamma]
    if df_gamma.empty:
        return
        
    greedy_runs = df_gamma[df_gamma['method'] == 'greedy']
    kamis_runs = df_gamma[df_gamma['method'] == 'kamis']
    
    if greedy_runs.empty or kamis_runs.empty:
        return
        
    greedy_agg = greedy_runs.groupby(['instance_idx', 'graph_type'])['l1_gradient_sum'].max().reset_index()
    greedy_agg.rename(columns={'l1_gradient_sum': 'greedy_val'}, inplace=True)
    
    kamis_agg = kamis_runs.groupby(['instance_idx', 'graph_type'])['l1_gradient_sum'].max().reset_index()
    kamis_agg.rename(columns={'l1_gradient_sum': 'kamis_val'}, inplace=True)
    
    merged = pd.merge(greedy_agg, kamis_agg, on=['instance_idx', 'graph_type'])
    
    if merged.empty:
        return
        
    g_val = merged['greedy_val']
    k_val = merged['kamis_val']
    # Calculate Wins (Tolerance of 1e-6)
    tolerance = 1e-6
    kamis_wins = sum((k_val - g_val) > tolerance)
    greedy_wins = sum((g_val - k_val) > tolerance)
    ties = sum(abs(k_val - g_val) <= tolerance)
    
    legend_label = f'y=x (Tie)\nKaMIS Wins: {kamis_wins}\nGreedy Wins: {greedy_wins}\nTies (±1e-6): {ties}'
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    balanced_merged = merged[merged['graph_type'] == 'Balanced']
    triangle_merged = merged[merged['graph_type'] == 'Triangle']
    
    if not balanced_merged.empty:
        ax.scatter(balanced_merged['greedy_val'], balanced_merged['kamis_val'], alpha=0.7, color='blue', marker='o', edgecolor='k', label='Balanced')
    if not triangle_merged.empty:
        ax.scatter(triangle_merged['greedy_val'], triangle_merged['kamis_val'], alpha=0.7, color='orange', marker='^', edgecolor='k', label='Triangle')
    
    min_val = min(g_val.min(), k_val.min()) - 0.01 if not g_val.empty else 0
    max_val = max(g_val.max(), k_val.max()) + 0.01 if not g_val.empty else 1
    if min_val == max_val: 
        min_val -= 0.1
        max_val += 0.1
        
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', label=legend_label)
    
    # Draw tolerance boundary
    import numpy as np
    x_vals = np.array([min_val, max_val])
    ax.fill_between(x_vals, x_vals - tolerance, x_vals + tolerance, color='red', alpha=0.1, label='Tie Region')
    
    ax.set_xlim([min_val, max_val])
    ax.set_ylim([min_val, max_val])
    ax.set_xlabel('Greedy L=1 Gradient Sum')
    ax.set_ylabel('KaMIS L=1 Gradient Sum')
    ax.set_title(f'L=1 Selected Operators Gradient Sum: KaMIS vs Greedy (Gamma = {gamma})')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='lower right')
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

def plot_layer_statistics_table(df, gamma, pdf):
    """
    Plots a text table showing the Mean, Median, Min, and Max Approximation Ratios 
    for Greedy and KaMIS at L=1, L=2, and L=3.
    """
    df_gamma = df[df['initial_gamma'] == gamma]
    if df_gamma.empty:
        return
        
    greedy_runs = df_gamma[df_gamma['method'] == 'greedy']
    kamis_runs = df_gamma[df_gamma['method'] == 'kamis']
    
    if greedy_runs.empty or kamis_runs.empty:
        return
        
    stats = []
    
    for layer in [1, 2, 3]:
        col = f'approx_l{layer}'
        
        for g_type in ['Balanced', 'Triangle']:
            g_runs_td = greedy_runs[greedy_runs['graph_type'] == g_type]
            k_runs_td = kamis_runs[kamis_runs['graph_type'] == g_type]
            if g_runs_td.empty or k_runs_td.empty:
                continue
                
            # Merge on instance to pair them perfectly for wilcoxon test
            g_data = g_runs_td.groupby('instance_idx')[col].max().rename('greedy_val')
            k_data = k_runs_td.groupby('instance_idx')[col].max().rename('kamis_val')
            merged = pd.merge(g_data, k_data, left_index=True, right_index=True)
            
            if merged.empty:
                continue
            
            g_mean, g_med, g_min, g_max = merged['greedy_val'].mean(), merged['greedy_val'].median(), merged['greedy_val'].min(), merged['greedy_val'].max()
            k_mean, k_med, k_min, k_max = merged['kamis_val'].mean(), merged['kamis_val'].median(), merged['kamis_val'].min(), merged['kamis_val'].max()
            
            # Wilcoxon Test
            try:
                diffs = merged['kamis_val'] - merged['greedy_val']
                # if all differences are perfectly zero, p=1.0 mathematically and scipy throws warning/exception
                if np.all(diffs == 0):
                    p_val_str = "1.0000"
                else:
                    stat, p_val = wilcoxon(merged['greedy_val'], merged['kamis_val'])
                    p_val_str = f"{p_val:.4e}"
            except Exception:
                p_val_str = "N/A"
                
            stats.append([f"L={layer} Greedy ({g_type})", f"{g_mean:.4f}", f"{g_med:.4f}", f"{g_min:.4f}", f"{g_max:.4f}", ""])
            stats.append([f"L={layer} KaMIS ({g_type})", f"{k_mean:.4f}", f"{k_med:.4f}", f"{k_min:.4f}", f"{k_max:.4f}", ""])
            stats.append([f"L={layer} Wilcoxon p-val ({g_type})", "", "", "", "", p_val_str])
        
        # Add a blank spacer row after each layer pairing except the last
        if layer != 3:
            stats.append(["", "", "", "", "", ""])

    table_data = [["Method", "Mean Approx", "Median Approx", "Min Approx", "Max Approx", "Wilcoxon P-val"]] + stats
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#d3d3d3')
        elif str(cell.get_text().get_text()).startswith("L="):
            cell.set_text_props(weight='bold')
            if "Greedy" in cell.get_text().get_text():
                cell.set_facecolor('#e6f2ff') # Light blue for greedy
            elif "KaMIS" in cell.get_text().get_text():
                cell.set_facecolor('#ebf5eb') # Light green for kamis
            
    plt.title(f"Layer-by-Layer Approximation Ratio Statistics (Gamma = {gamma})", fontsize=14, fontweight='bold', y=0.95)
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

def plot_composition_table(df_comp, gamma, pdf):
    """
    Plots a text table summarizing the operator composition for the given gamma.
    """
    if df_comp is None or df_comp.empty:
        return
        
    df_gamma = df_comp[df_comp['gamma'] == gamma]
    if df_gamma.empty:
        return
        
    # We only care about L=1, L=2, L=3 for the report
    stats = []
    
    for layer in [1, 2, 3]:
        df_l = df_gamma[df_gamma['layer'] == layer]
        if df_l.empty:
            continue
            
        g_data = df_l[df_l['method'] == 'greedy']
        k_data = df_l[df_l['method'] == 'kamis']
        
        g_mix = g_data['global_qaoa_mixers'].mean() if not g_data.empty else 0
        g_q1 = g_data['1_qubit_ops'].mean() if not g_data.empty else 0
        g_q2 = g_data['2_qubit_ops'].mean() if not g_data.empty else 0
        g_tot = g_data['total_ops'].mean() if not g_data.empty else 0
        g_dens = g_data['layer_density'].mean() if not g_data.empty else 0
        g_disj = str(g_data['is_disjoint'].all()) if not g_data.empty else "N/A"
        
        k_mix = k_data['global_qaoa_mixers'].mean() if not k_data.empty else 0
        k_q1 = k_data['1_qubit_ops'].mean() if not k_data.empty else 0
        k_q2 = k_data['2_qubit_ops'].mean() if not k_data.empty else 0
        k_tot = k_data['total_ops'].mean() if not k_data.empty else 0
        k_dens = k_data['layer_density'].mean() if not k_data.empty else 0
        k_disj = str(k_data['is_disjoint'].all()) if not k_data.empty else "N/A"
        
        stats.append([f"L={layer} Greedy", f"{g_tot:.2f}", f"{g_mix:.2f}", f"{g_q1:.2f}", f"{g_q2:.2f}", f"{g_dens:.2f}", g_disj])
        stats.append([f"L={layer} KaMIS", f"{k_tot:.2f}", f"{k_mix:.2f}", f"{k_q1:.2f}", f"{k_q2:.2f}", f"{k_dens:.2f}", k_disj])
        
        if layer != 3:
            stats.append(["", "", "", "", "", "", ""])

    if not stats:
        return
            
    table_data = [["Method", "Avg Total Ops", "Avg Global QAOA Mixer", "Avg 1-Qubit (X/Y/Z)", "Avg 2-Qubit", "Avg Density", "All Disjoint?"]] + stats
    
    fig, ax = plt.subplots(figsize=(14, 3.5))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#d3d3d3')
        elif str(cell.get_text().get_text()).startswith("L="):
            cell.set_text_props(weight='bold')
            if "Greedy" in cell.get_text().get_text():
                cell.set_facecolor('#e6f2ff')
            elif "KaMIS" in cell.get_text().get_text():
                cell.set_facecolor('#ebf5eb')
                
    plt.title(f"Average Operator Selection Composition (Gamma = {gamma})", fontsize=14, fontweight='bold', y=0.95)
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

def plot_approx_ratio_boxplots(df, gamma, pdf):
    """
    Plots stratified box plots for the approximation ratio at L=1, L=2, and L=3.
    Stratified by Gamma (already filtered), Layer (x-axis), Graph Type (subplots), and Method (hue).
    """
    df_gamma = df[df['initial_gamma'] == gamma]
    if df_gamma.empty:
        return
        
    # We need to collect data for L=1, 2, 3 into a long format for seaborn
    records = []
    
    for layer in [1, 2, 3]:
        col = f'approx_l{layer}'
        
        # Get max approx per instance, layer, method, and graph type
        for method in ['greedy', 'kamis']:
            m_runs = df_gamma[df_gamma['method'] == method]
            if m_runs.empty:
                continue
                
            for g_type in ['Balanced', 'Triangle']:
                g_runs = m_runs[m_runs['graph_type'] == g_type]
                if g_runs.empty:
                    continue
                    
                agg = g_runs.groupby('instance_idx')[col].max().reset_index()
                
                for _, row in agg.iterrows():
                    val = row[col]
                    if pd.notna(val):
                        records.append({
                            'Layer': f"L={layer}",
                            'Approximation Ratio': val,
                            'Method': 'Greedy' if method == 'greedy' else 'KaMIS',
                            'Graph Type': g_type
                        })
                        
    if not records:
        return
        
    df_box = pd.DataFrame(records)
    
    # We will create one figure with 2 subplots (1 for Balanced, 1 for Triangle)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    
    # Check if we have data for each type to avoid plotting errors
    has_balanced = not df_box[df_box['Graph Type'] == 'Balanced'].empty
    has_triangle = not df_box[df_box['Graph Type'] == 'Triangle'].empty
    
    if has_balanced:
        sns.boxplot(ax=axes[0], data=df_box[df_box['Graph Type'] == 'Balanced'], 
                    x='Layer', y='Approximation Ratio', hue='Method', 
                    palette={'Greedy': '#aec7e8', 'KaMIS': '#98df8a'}, showmeans=True,
                    meanprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"black"})
        axes[0].set_title('Balanced Instances')
        axes[0].grid(True, linestyle='--', alpha=0.5, axis='y')
    else:
        axes[0].text(0.5, 0.5, 'No Balanced Data', ha='center', va='center')
        
    if has_triangle:
        sns.boxplot(ax=axes[1], data=df_box[df_box['Graph Type'] == 'Triangle'], 
                    x='Layer', y='Approximation Ratio', hue='Method', 
                    palette={'Greedy': '#aec7e8', 'KaMIS': '#98df8a'}, showmeans=True,
                    meanprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"black"})
        axes[1].set_title('Triangle Instances')
        axes[1].grid(True, linestyle='--', alpha=0.5, axis='y')
    else:
        axes[1].text(0.5, 0.5, 'No Triangle Data', ha='center', va='center')
        
    # Adjust overall title
    fig.suptitle(f"Approximation Ratio Distributions: KaMIS vs Greedy (Gamma = {gamma})", fontsize=16, fontweight='bold')
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Generate Layer-Wise Approximation Ratio Comparison Scatters")
    parser.add_argument("results_dir", help="Directory containing the benchmark JSON files.")
    parser.add_argument("--output", default="layer_analysis_summary", help="Base name for the output files (without extension).")
    args = parser.parse_args()

    results_dir = args.results_dir
    output_base = args.output
    output_pdf = os.path.join(results_dir, f"{output_base}.pdf")
    output_csv = os.path.join(results_dir, f"{output_base}.csv")

    print(f"Loading data from {results_dir}...")
    df = load_all_data(results_dir)
    
    if df.empty:
        print("No valid data found in _all_ files for plotting.")
        return
        
    # Save the parsed matrix to CSV for easy spreadsheet inspection
    df.to_csv(output_csv, index=False)
    print(f"Layer-wise raw data saved to {output_csv}")
    
    # Also load the composition data so we can add it to the report
    try:
        print("Parsing operator maps to determine selection compositions...")
        df_comp = get_operator_composition_data(results_dir)
    except Exception as e:
        print(f"Warning: Could not compile operator compositions: {e}")
        df_comp = None
    
    unique_gammas = sorted(df['initial_gamma'].dropna().unique())
    print(f"Found gammas: {unique_gammas}")
    
    with PdfPages(output_pdf) as pdf:
        for gamma in unique_gammas:
            print(f"Generating stats table, composition table, boxplots, L=1 Gradient scatter, and L=1-3 ratio scatters for Gamma = {gamma}")
            plot_layer_statistics_table(df, gamma, pdf=pdf)
            plot_composition_table(df_comp, gamma, pdf=pdf)
            plot_approx_ratio_boxplots(df, gamma, pdf=pdf)
            plot_l1_gradient_scatter(df, gamma, pdf=pdf)
            plot_layer_scatters(df, gamma, layer_depth=1, pdf=pdf)
            plot_layer_scatters(df, gamma, layer_depth=2, pdf=pdf)
            plot_layer_scatters(df, gamma, layer_depth=3, pdf=pdf)
            
    print(f"PDF Report saved to {output_pdf}")

if __name__ == "__main__":
    main()
