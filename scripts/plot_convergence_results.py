
import argparse
import json
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import seaborn as sns
import os
import glob
from datetime import datetime
import numpy as np

def load_data(results_dir):
    """
    Loads data from a directory containing both _all_ and _best_ JSON files.
    Returns two DataFrames: df_all and df_best.
    """
    all_files = glob.glob(os.path.join(results_dir, "benchmark_*_all_*.json"))
    best_files = glob.glob(os.path.join(results_dir, "benchmark_*_best_*.json"))

    if not all_files:
        print(f"Warning: No '_all_' files found in {results_dir}")
    if not best_files:
        print(f"Warning: No '_best_' files found in {results_dir}")

    def load_files(file_list):
        records = []
        for fpath in file_list:
            try:
                with open(fpath, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, dict) and "results" in data:
                        records.extend(data["results"])
                    elif isinstance(data, list):
                        records.extend(data)
            except Exception as e:
                print(f"Error loading {fpath}: {e}")
        return pd.DataFrame(records)

    df_all = load_files(all_files)
    df_best = load_files(best_files)

    return df_all, df_best

def reformat_config_label(label):
    """
    Reformats the configuration label from e.g. 'Greedy_NewPool_g0.001' 
    to 'greedy; g = 0.001' or 'kamis; g = 0.001'.
    """
    if not isinstance(label, str):
        return label
    
    label_lower = label.lower()
    if 'greedy' in label_lower:
        method = 'greedy'
    elif 'kamis' in label_lower:
        method = 'kamis'
    else:
        return label
        
    if '_g' in label:
        gamma = label.split('_g')[-1]
        return f"{method}; g = {gamma}"
        
    return label

def create_summary_table(df_all, df_best):
    """
    Generates a summary DataFrame with Mean, Median, Max, Min for:
    - Time (adapt_time)
    - Layers (layers)
    - Satisfaction (tetris_satisfaction_percent)
    
    Rows: Each configuration + "Best Instances Per Configuration"
    """
    stats_cols = ['adapt_time', 'layers', 'tetris_satisfaction_percent']
    
    # 1. Process "All" Configurations
    # Group by config_label
    summary_rows = []
    
    if not df_all.empty:
        grouped = df_all.groupby('config_label')
        for name, group in grouped:
            row = {'Configuration': name}
            for col in stats_cols:
                row[f'{col} (Mean)'] = group[col].mean()
                row[f'{col} (Median)'] = group[col].median()
                row[f'{col} (Max)'] = group[col].max()
                if col == 'tetris_satisfaction_percent':
                    row[f'{col} (Min)'] = group[col].min()
            summary_rows.append(row)

    # 2. Process "Best" Configuration (The Hybrid/Oracle result)
    if not df_best.empty:
        row = {'Configuration': 'Best Result'}
        for col in stats_cols:
            row[f'{col} (Mean)'] = df_best[col].mean()
            row[f'{col} (Median)'] = df_best[col].median()
            row[f'{col} (Max)'] = df_best[col].max()
            if col == 'tetris_satisfaction_percent':
                row[f'{col} (Min)'] = df_best[col].min()
        summary_rows.append(row)
        
    summary_df = pd.DataFrame(summary_rows)
    
    # Reorder columns for clarity
    # Config | Sat (Mean, Med, Max, Min) | Layers (Mean, Med, Max) | Time (Mean, Med, Max)
    ordered_cols = ['Configuration']
    labels = ['Satisfaction', 'Layers', 'Time']
    orig_cols = ['tetris_satisfaction_percent', 'layers', 'adapt_time']
    
    final_cols = ['Configuration']
    for label, orig in zip(labels, orig_cols):
        final_cols.append(f'{orig} (Mean)')
        final_cols.append(f'{orig} (Median)')
        final_cols.append(f'{orig} (Max)')
        if orig == 'tetris_satisfaction_percent':
            final_cols.append(f'{orig} (Min)')
        
    # Filter only existing columns
    final_cols = [c for c in final_cols if c in summary_df.columns]
    
    return summary_df[final_cols]

def create_approx_ratio_table(df_all, df_best):
    """
    Generates a summary DataFrame with Mean, Median, Min, Max for the Approximation Ratio:
    Ratio = tetris_satisfaction_percent / (max_satisfied_gurobi / num_clauses)
    """
    stats_col = 'approx_ratio'
    summary_rows = []
    
    def calculate_ratio(df):
        df = df.copy()
        if 'max_satisfied_gurobi' in df.columns and 'num_clauses' in df.columns:
            gurobi_sat_percent = df['max_satisfied_gurobi'] / df['num_clauses']
            # Avoid division by zero
            gurobi_sat_percent = gurobi_sat_percent.replace(0, 1e-10)
            df[stats_col] = df['tetris_satisfaction_percent'] / gurobi_sat_percent
        else:
            df[stats_col] = float('nan')
        return df

    # 1. Process "All" Configurations
    df_all_ratio = calculate_ratio(df_all)
    if not df_all_ratio.empty:
        grouped = df_all_ratio.groupby('config_label')
        for name, group in grouped:
            row = {'Configuration': name}
            valid_ratios = group[stats_col].dropna()
            if not valid_ratios.empty:
                row[f'{stats_col} (Mean)'] = valid_ratios.mean()
                row[f'{stats_col} (Median)'] = valid_ratios.median()
                row[f'{stats_col} (Min)'] = valid_ratios.min()
                row[f'{stats_col} (Max)'] = valid_ratios.max()
            summary_rows.append(row)

    # 2. Process "Best" Configuration
    df_best_ratio = calculate_ratio(df_best)
    if not df_best_ratio.empty:
        row = {'Configuration': 'Best Result'}
        valid_ratios = df_best_ratio[stats_col].dropna()
        if not valid_ratios.empty:
            row[f'{stats_col} (Mean)'] = valid_ratios.mean()
            row[f'{stats_col} (Median)'] = valid_ratios.median()
            row[f'{stats_col} (Min)'] = valid_ratios.min()
            row[f'{stats_col} (Max)'] = valid_ratios.max()
        summary_rows.append(row)
        
    summary_df = pd.DataFrame(summary_rows)
    return summary_df

def plot_outlier_counts(df_all, df_best, pdf, title_suffix=""):
    """
    Plots a text page listing the instance IDs that are above 2 std deviations of layers median
    and 2 std deviations below the approx_ratio mean, ONLY for the Best Result.
    """
    if df_best.empty or len(df_best) < 2:
        return
        
    df = df_best.copy()
    if 'max_satisfied_gurobi' in df.columns and 'num_clauses' in df.columns:
        df['gurobi_sat_percent'] = df['max_satisfied_gurobi'] / df['num_clauses']
        df['gurobi_sat_percent'] = df['gurobi_sat_percent'].replace(0, 1e-10)
        df['approx_ratio'] = df['tetris_satisfaction_percent'] / df['gurobi_sat_percent']
    else:
        print("Missing gurobi or clauses data for outlier approx_ratio calculation.")
        return
        
    layer_median = df['layers'].median()
    layer_std = df['layers'].std()
    approx_mean = df['approx_ratio'].mean()
    approx_std = df['approx_ratio'].std()
    
    layer_threshold = layer_median + (2 * layer_std)
    approx_threshold = approx_mean - (2 * approx_std)
    
    high_layers_df = df[df['layers'] > layer_threshold]
    low_approx_df = df[df['approx_ratio'] < approx_threshold]
    
    high_layers_ids = high_layers_df['instance_idx'].tolist() if 'instance_idx' in high_layers_df.columns else []
    low_approx_ids = low_approx_df['instance_idx'].tolist() if 'instance_idx' in low_approx_df.columns else []
    
    high_layers_info = [str(idx) for idx in high_layers_ids]
    low_approx_info = [str(idx) for idx in low_approx_ids]

    print("\n" + "="*80)
    print(f"Outlier Analysis for Best Instances selected by Grid Search - {title_suffix}")
    print("="*80)
    print(f"High Layers (ID: Val): {high_layers_info}")
    print(f"Low Approx Ratio (ID: Val): {low_approx_info}")
    print("="*80 + "\n")
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis('off')
    
    plt.title(f"Outlier Analysis for Best Instances selected by Grid Search {title_suffix}", fontsize=12, fontweight='bold', y=0.95)
    
    text_content = (
        f"Layers:             median = {layer_median:.2f},  std = {layer_std:.2f}  "
        f"(threshold (median + 2sd) = {layer_threshold:.2f})\n"
        f"Approx Ratio:    mean   = {approx_mean:.4f},  std = {approx_std:.4f}  "
        f"(threshold (mean - 2sd) = {approx_threshold:.4f})\n\n"
        f"Instance IDs with Layers > Threshold:\n"
        f"{', '.join(high_layers_info) if high_layers_info else 'None'}\n\n"
        f"Instance IDs with Approx Ratio < Threshold:\n"
        f"{', '.join(low_approx_info) if low_approx_info else 'None'}"
    )
    
    ax.text(0.1, 0.8, text_content, 
            transform=ax.transAxes, 
            fontsize=12, 
            verticalalignment='top', 
            wrap=True)
            
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

def plot_best_config_distribution(df_best, pdf, title_suffix=""):
    """
    Plots the distribution of configurations that achieved the 'best' result.
    Adds the plot to the provided PDF object.
    """
    if df_best.empty:
        return

    # Count occurrences of each config_label
    config_counts = df_best['config_label'].value_counts()
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x=config_counts.values, y=config_counts.index, hue=config_counts.index, palette="viridis", legend=False)
    
    plt.title(f"Distribution of Best Configurations {title_suffix}")
    plt.xlabel("Number of Instances")
    plt.ylabel("Configuration")
    plt.tight_layout()
    
    # Add counts to the end of bars
    for i, v in enumerate(config_counts.values):
        plt.text(v + 0.1, i, str(v), color='black', va='center')

    pdf.savefig()
    plt.close()

def plot_time_histogram(df_best, pdf, title_suffix=""):
    """
    Plots a histogram of the convergence times for the best results.
    """
    if df_best.empty or 'adapt_time' not in df_best.columns:
        return

    plt.figure(figsize=(10, 6))
    
    # sns.histplot automatically creates the histogram bins and optionally a KDE curve
    sns.histplot(data=df_best, x='adapt_time', bins=20, kde=True, color='blue')
    
    plt.title(f"Histogram of Convergence Times (Best Results) {title_suffix}")
    plt.xlabel("Convergence Time (s)")
    plt.ylabel("Number of Instances")
    plt.tight_layout()
    
    pdf.savefig()
    plt.close()

def plot_layers_histogram(df_best, pdf, title_suffix=""):
    """
    Plots a histogram of the number of layers at convergence for the best results.
    """
    if df_best.empty or 'layers' not in df_best.columns:
        return

    plt.figure(figsize=(10, 6))
    
    sns.histplot(data=df_best, x='layers', discrete=True, kde=False, color='purple')
    
    # Force x-axis to display only integers
    ax = plt.gca()
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
    plt.title(f"Histogram of Number of Layers at Convergence (Best Results) {title_suffix}")
    plt.xlabel("Number of Layers")
    plt.ylabel("Number of Instances")
    plt.tight_layout()
    
    pdf.savefig()
    plt.close()

def plot_config_parameters_table(df_all, pdf, title_suffix=""):
    """
    Extracts static configuration parameters from the DataFrame and plots them as a table.
    """
    if df_all.empty:
        return

    # List of columns to extract if they exist
    config_keys = [
        'hamiltonian_type', 'num_shots', 'layer_stopper_max', 
        'floor_stopper_threshold', 'optimizer_tolerance', 'optimizer_max_iterations', 
        'slow_stopper_threshold', 'slow_stopper_patience', 'gradient_threshold', 
        'score_stopper_threshold', 'percent_tail_ends_removed'
    ]

    # Use the first row to determine the parameters since they are static across jobs
    first_row = df_all.iloc[0]
    config_dict = {}
    
    for key in config_keys:
        if key in first_row:
            config_dict[key] = first_row[key]

    if not config_dict:
        return

    # Create a DataFrame for the table
    config_df = pd.DataFrame(list(config_dict.items()), columns=['Parameter', 'Value'])

    fig, ax = plt.subplots(figsize=(8, len(config_df) * 0.4 + 1))
    ax.axis('tight')
    ax.axis('off')

    table = ax.table(cellText=config_df.values,
                     colLabels=config_df.columns,
                     loc='center',
                     cellLoc='left')

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    plt.title(f"Configuration Parameters {title_suffix}", fontsize=14, fontweight='bold', y=0.95)

    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

def plot_tie_statistics_table(df_all, df_best, pdf, title_suffix=""):
    """
    Plots a table showing how many times Greedy won, KaMIS won, or they tied
    head-to-head across all instances.
    """
    if df_all.empty: return
    
    greedy_pure_wins = 0
    kamis_pure_wins = 0
    ties = 0
    
    # Analyze head-to-head performance on each unique instance
    unique_instances = df_all['instance_idx'].dropna().unique()
    
    for iid in unique_instances:
        inst_runs = df_all[df_all['instance_idx'] == iid]
        
        # Split runs for this instance into Greedy and KaMIS
        greedy_runs = inst_runs[inst_runs['config_label'].astype(str).str.lower().str.contains('greedy')]
        kamis_runs = inst_runs[inst_runs['config_label'].astype(str).str.lower().str.contains('kamis')]
        
        if greedy_runs.empty or kamis_runs.empty: 
            continue
            
        # Find the absolute best run for Greedy on this instance (Max Sat, Min Layers)
        greedy_best = greedy_runs.sort_values(by=['tetris_satisfaction_percent', 'layers'], ascending=[False, True]).iloc[0]
        kamis_best = kamis_runs.sort_values(by=['tetris_satisfaction_percent', 'layers'], ascending=[False, True]).iloc[0]
        
        g_sat = greedy_best['tetris_satisfaction_percent']
        g_lay = greedy_best['layers']
        
        k_sat = kamis_best['tetris_satisfaction_percent']
        k_lay = kamis_best['layers']
        
        # Head-to-head breakdown
        if g_sat > k_sat:
            greedy_pure_wins += 1
        elif k_sat > g_sat:
            kamis_pure_wins += 1
        else:
            # Satisfaction is identical, check layers
            if g_lay < k_lay:
                greedy_pure_wins += 1
            elif k_lay < g_lay:
                kamis_pure_wins += 1
            else:
                ties += 1

    table_data = [
        ["Metric", "Count"],
        ["Greedy Pure Wins", str(greedy_pure_wins)],
        ["KaMIS Pure Wins", str(kamis_pure_wins)],
        ["Exact Ties", str(ties)],
    ]
    
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=table_data,
                     loc='center',
                     cellLoc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#d3d3d3')
            
    plt.title(f"Greedy vs KaMIS Tie-Breaking Statistics {title_suffix}", fontsize=14, fontweight='bold', y=0.95)
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

def plot_gamma_stratified_table(df_all, pdf, title_suffix=""):
    """
    Creates a side-by-side comparison table of KaMIS vs Greedy across all initial_gamma values.
    """
    if df_all.empty: return
    
    df = df_all.copy()
    if 'max_satisfied_gurobi' in df.columns and 'num_clauses' in df.columns:
        df['gurobi_sat_percent'] = df['max_satisfied_gurobi'] / df['num_clauses']
        df['gurobi_sat_percent'] = df['gurobi_sat_percent'].replace(0, 1e-10)
        df['approx_ratio'] = df['tetris_satisfaction_percent'] / df['gurobi_sat_percent']
    else:
        df['approx_ratio'] = np.nan
        
    df['method'] = df['config_label'].apply(lambda x: 'kamis' if 'kamis' in str(x).lower() else ('greedy' if 'greedy' in str(x).lower() else str(x)))
    
    def get_gamma(lbl):
        try:
            return float(lbl.split('g = ')[-1])
        except:
            if '_g' in str(lbl):
                try:
                    return float(lbl.split('_g')[-1])
                except:
                    pass
            return np.nan
            
    df['gamma'] = df['config_label'].apply(get_gamma)
    
    summary_data = []
    gammas = sorted([g for g in df['gamma'].unique() if pd.notna(g)])
    
    for g in gammas:
        g_df = df[df['gamma'] == g]
        kamis_df = g_df[g_df['method'] == 'kamis']
        greedy_df = g_df[g_df['method'] == 'greedy']
        
        row = {'Initial Gamma': g}
        row['KaMIS Approx Ratio'] = kamis_df['approx_ratio'].mean() if not kamis_df.empty else np.nan
        row['Greedy Approx Ratio'] = greedy_df['approx_ratio'].mean() if not greedy_df.empty else np.nan
        row['KaMIS Avg Layers'] = kamis_df['layers'].mean() if not kamis_df.empty else np.nan
        row['Greedy Avg Layers'] = greedy_df['layers'].mean() if not greedy_df.empty else np.nan
        
        summary_data.append(row)
        
    if not summary_data: return
    
    table_df = pd.DataFrame(summary_data)
    table_df = table_df.round(4)
    
    fig, ax = plt.subplots(figsize=(10, len(table_df)*0.5 + 2))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=table_df.values,
                     colLabels=table_df.columns,
                     loc='center',
                     cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#d3d3d3')
            
    plt.title(f"Gamma-Stratified Comparison {title_suffix}", fontsize=14, fontweight='bold', y=0.95)
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

def plot_scatter_comparisons(df_all, pdf, title_suffix=""):
    """
    Creates scatter plots (Approx Ratio and Layers) comparing KaMIS and Greedy for each gamma.
    """
    if df_all.empty: return
    df = df_all.copy()
    
    if 'max_satisfied_gurobi' in df.columns and 'num_clauses' in df.columns:
        df['gurobi_sat_percent'] = df['max_satisfied_gurobi'] / df['num_clauses']
        df['gurobi_sat_percent'] = df['gurobi_sat_percent'].replace(0, 1e-10)
        df['approx_ratio'] = df['tetris_satisfaction_percent'] / df['gurobi_sat_percent']
    else:
        df['approx_ratio'] = np.nan
        
    df['method'] = df['config_label'].apply(lambda x: 'kamis' if 'kamis' in str(x).lower() else ('greedy' if 'greedy' in str(x).lower() else str(x)))
    
    def get_gamma(lbl):
        try:
            return float(lbl.split('g = ')[-1])
        except:
            if '_g' in str(lbl):
                try:
                    return float(lbl.split('_g')[-1])
                except:
                    pass
            return np.nan
            
    df['gamma'] = df['config_label'].apply(get_gamma)
    gammas = sorted([g for g in df['gamma'].unique() if pd.notna(g)])
    
    for g in gammas:
        g_df = df[df['gamma'] == g]
        kamis_df = g_df[g_df['method'] == 'kamis'].set_index('instance_idx')
        greedy_df = g_df[g_df['method'] == 'greedy'].set_index('instance_idx')
        
        # intersect common instances
        common_idx = kamis_df.index.intersection(greedy_df.index)
        if common_idx.empty: continue
        
        k_approx = kamis_df.loc[common_idx, 'approx_ratio']
        g_approx = greedy_df.loc[common_idx, 'approx_ratio']
        
        k_layers = kamis_df.loc[common_idx, 'layers']
        g_layers = greedy_df.loc[common_idx, 'layers']
        
        # Calculate Wins for Approx Ratio (Higher is better)
        approx_kamis_wins = sum(k_approx > g_approx)
        approx_greedy_wins = sum(g_approx > k_approx)
        approx_ties = sum(k_approx == g_approx)
        
        approx_legend_label = f'y=x (Tie)\nKaMIS Wins: {approx_kamis_wins}\nGreedy Wins: {approx_greedy_wins}\nTies: {approx_ties}'
        
        # Plot Approx Ratio Scatter
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(g_approx, k_approx, alpha=0.7, color='blue', edgecolor='k')
        
        min_val = min(g_approx.min(), k_approx.min()) - 0.01 if not g_approx.empty else 0
        max_val = max(g_approx.max(), k_approx.max()) + 0.01 if not g_approx.empty else 1
        if min_val == max_val: 
            min_val -= 0.1
            max_val += 0.1
            
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label=approx_legend_label)
        
        ax.set_xlim([min_val, max_val])
        ax.set_ylim([min_val, max_val])
        ax.set_xlabel('Greedy Approx Ratio')
        ax.set_ylabel('KaMIS Approx Ratio')
        ax.set_title(f'Approx Ratio: KaMIS vs Greedy (Gamma = {g}) {title_suffix}')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(loc='lower right')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Calculate Wins for Layers (Lower is better)
        layers_kamis_wins = sum(k_layers < g_layers)
        layers_greedy_wins = sum(g_layers < k_layers)
        layers_ties = sum(k_layers == g_layers)
        
        layers_legend_label = f'y=x (Tie)\nKaMIS Wins: {layers_kamis_wins}\nGreedy Wins: {layers_greedy_wins}\nTies: {layers_ties}'
        
        # Plot Layers Scatter
        fig, ax = plt.subplots(figsize=(8, 8))
        jitter_g = np.random.uniform(-0.2, 0.2, size=len(g_layers))
        jitter_k = np.random.uniform(-0.2, 0.2, size=len(k_layers))
        ax.scatter(g_layers + jitter_g, k_layers + jitter_k, alpha=0.5, color='orange', edgecolor='k')
        
        min_l = min(g_layers.min(), k_layers.min()) - 1 if not g_layers.empty else 0
        max_l = max(g_layers.max(), k_layers.max()) + 1 if not g_layers.empty else 1
        if min_l == max_l:
            min_l -= 1
            max_l += 1
            
        ax.plot([min_l, max_l], [min_l, max_l], 'r--', label=layers_legend_label)
        
        ax.set_xlim([min_l, max_l])
        ax.set_ylim([min_l, max_l])
        ax.set_xlabel('Greedy Num Layers (with jitter)')
        ax.set_ylabel('KaMIS Num Layers (with jitter)')
        ax.set_title(f'Number of Layers: KaMIS vs Greedy (Gamma = {g}) {title_suffix}')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(loc='lower right')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="Generate Summary Table from Benchmark Results")
    parser.add_argument("results_dir", help="Directory containing the benchmark JSON files (timestamped folder)")
    parser.add_argument("--output", default="benchmark_summary", help="Output filename base (without extension)")
    args = parser.parse_args()

    print(f"Loading data from {args.results_dir}...")
    df_all, df_best = load_data(args.results_dir)
    
    if not df_all.empty:
        df_all['config_label'] = df_all['config_label'].apply(reformat_config_label)
    if not df_best.empty:
        df_best['config_label'] = df_best['config_label'].apply(reformat_config_label)
    
    print(f"Loaded {len(df_all)} records for individual configs.")
    print(f"Loaded {len(df_best)} records for best instances.")

    if df_all.empty and df_best.empty:
        print("No data found. Exiting.")
        return

    # Determine unique types for stratification
    types = ["All"]
    if not df_best.empty and "type" in df_best.columns:
        types.extend(sorted([str(t) for t in df_best["type"].unique() if pd.notna(t)]))

    for t in types:
        if t == "All":
            t_df_all = df_all
            t_df_best = df_best
            out_prefix = args.output
            title_suffix = ""
            print_prefix = "All Instances"
        else:
            t_df_all = df_all[df_all["type"] == t] if not df_all.empty and "type" in df_all.columns else pd.DataFrame()
            t_df_best = df_best[df_best["type"] == t]
            out_prefix = f"{args.output}_{t}"
            title_suffix = f"({t.capitalize()} Instances)"
            print_prefix = f"{t.capitalize()} Instances"

        if t_df_best.empty:
            continue

        print(f"\nProcessing: {print_prefix}")

        # Create Summary Table
        summary_table = create_summary_table(t_df_all, t_df_best)
        approx_ratio_table = create_approx_ratio_table(t_df_all, t_df_best)
        
        rename_map = {
            'adapt_time (Mean)': 'Time Mean (s)',
            'adapt_time (Median)': 'Time Med (s)',
            'adapt_time (Max)': 'Time Max (s)',
            'layers (Mean)': 'Layers Mean',
            'layers (Median)': 'Layers Med',
            'layers (Max)': 'Layers Max',
            'tetris_satisfaction_percent (Mean)': 'Sat % Mean',
            'tetris_satisfaction_percent (Median)': 'Sat % Med',
            'tetris_satisfaction_percent (Max)': 'Sat % Max',
            'tetris_satisfaction_percent (Min)': 'Sat % Min'
        }
        summary_table = summary_table.rename(columns=rename_map)

        # Round numeric columns
        numeric_cols = summary_table.select_dtypes(include=[np.number]).columns
        summary_table[numeric_cols] = summary_table[numeric_cols].round(4)

        # Save to CSV
        output_csv = f"{out_prefix}.csv"
        summary_table.to_csv(output_csv, index=False)
        print(f"Table saved to {output_csv}")

        # Generate PDF
        output_pdf = f"{out_prefix}.pdf"
        with PdfPages(output_pdf) as pdf:
            # Page 1: Summary Table
            fig, ax = plt.subplots(figsize=(12, len(summary_table) * 0.5 + 2))
            ax.axis('tight')
            ax.axis('off')
            
            table = ax.table(cellText=summary_table.values,
                             colLabels=summary_table.columns,
                             loc='center',
                             cellLoc='center')
            
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1.2, 1.2)
            
            # Adjust title position based on number of tables
            plt.title(f"Benchmark Summary: {os.path.basename(args.results_dir)} {title_suffix}\n\n", y=0.98)
            
            # Second Table: Approximation Ratio
            if not approx_ratio_table.empty and len(approx_ratio_table.columns) > 1:
                # Reformat approx ratio table columns for display
                approx_rename_map = {
                    'approx_ratio (Mean)': 'Approx Ratio Mean',
                    'approx_ratio (Median)': 'Approx Ratio Med',
                    'approx_ratio (Min)': 'Approx Ratio Min',
                    'approx_ratio (Max)': 'Approx Ratio Max'
                }
                approx_ratio_table = approx_ratio_table.rename(columns=approx_rename_map)
                
                # Round numeric columns
                approx_numeric_cols = approx_ratio_table.select_dtypes(include=[np.number]).columns
                approx_ratio_table[approx_numeric_cols] = approx_ratio_table[approx_numeric_cols].round(4)
                
                # Append rows visually to the same plot
                table2 = ax.table(cellText=approx_ratio_table.values,
                                 colLabels=approx_ratio_table.columns,
                                 loc='bottom',
                                 bbox=[0.0, -0.6, 1.0, 0.4], # [x0, y0, width, height]
                                 cellLoc='center')
                table2.auto_set_font_size(False)
                table2.set_fontsize(8)
                
                # Save the approximation ratio table to CSV as well
                output_approx_csv = f"{out_prefix}_approx_ratio.csv"
                approx_ratio_table.to_csv(output_approx_csv, index=False)
                print(f"Approximation Ratio Table saved to {output_approx_csv}")

            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
            # Page 2: Distribution Plot
            plot_best_config_distribution(t_df_best, pdf, title_suffix)
            
            # Page 3: Tie-Breaking Statistics
            plot_tie_statistics_table(t_df_all, t_df_best, pdf, title_suffix)
            
            # Page 4: Outlier Counts
            plot_outlier_counts(t_df_all, t_df_best, pdf, title_suffix)
            
            # Page 5: Convergence Time Histogram
            plot_time_histogram(t_df_best, pdf, title_suffix)

            # Page 6: Layers Histogram
            plot_layers_histogram(t_df_best, pdf, title_suffix)
            
            # Page 7: Configuration Parameters Table
            plot_config_parameters_table(t_df_all, pdf, title_suffix)
            
            # Page 8: Gamma-Stratified Comparison Table
            plot_gamma_stratified_table(t_df_all, pdf, title_suffix)
            
            # Page 9+: Scatter Plots for Approx Ratio and Layers
            plot_scatter_comparisons(t_df_all, pdf, title_suffix)
            
        print(f"PDF Report saved to {output_pdf}")

if __name__ == "__main__":
    main()
