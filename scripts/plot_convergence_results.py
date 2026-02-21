
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
        row = {'Configuration': 'Best Instances Per Configuration'}
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
        row = {'Configuration': 'Best Instances Per Configuration'}
        valid_ratios = df_best_ratio[stats_col].dropna()
        if not valid_ratios.empty:
            row[f'{stats_col} (Mean)'] = valid_ratios.mean()
            row[f'{stats_col} (Median)'] = valid_ratios.median()
            row[f'{stats_col} (Min)'] = valid_ratios.min()
            row[f'{stats_col} (Max)'] = valid_ratios.max()
        summary_rows.append(row)
        
    summary_df = pd.DataFrame(summary_rows)
    return summary_df

def plot_outlier_counts(df_all, df_best, pdf):
    """
    Plots a table of the number of instances that are above 1 std deviation of layers median
    and 1 std deviation below the sat mean.
    """
    rows = []
    
    def get_count(df, label):
        if df.empty or len(df) < 2:
            return {
                'Configuration': label, 
                'High Layers (> Med+1SD)': 0,
                'Low Sat (< Mean-1SD)': 0,
                'High Layers IDs': "",
                'Low Sat IDs': ""
            }
        layer_median = df['layers'].median()
        layer_std = df['layers'].std()
        sat_mean = df['tetris_satisfaction_percent'].mean()
        sat_std = df['tetris_satisfaction_percent'].std()
        
        layer_threshold = layer_median + layer_std
        sat_threshold = sat_mean - sat_std
        
        high_layers_df = df[df['layers'] > layer_threshold]
        count_layers = len(high_layers_df)
        
        low_sat_df = df[df['tetris_satisfaction_percent'] < sat_threshold]
        count_sat = len(low_sat_df)
        
        outliers_df = df[(df['layers'] > layer_threshold) & (df['tetris_satisfaction_percent'] < sat_threshold)]
        
        high_layers_ids = ""
        low_sat_ids = ""
        
        if count_layers > 0 and 'instance_idx' in high_layers_df.columns:
            high_layers_ids = ",".join(map(str, high_layers_df['instance_idx'].tolist()))
            
        if count_sat > 0 and 'instance_idx' in low_sat_df.columns:
            low_sat_ids = ",".join(map(str, low_sat_df['instance_idx'].tolist()))
        
        return {
            'Configuration': label, 
            'High Layers (> Med+1SD)': count_layers,
            'Low Sat (< Mean-1SD)': count_sat,
            'High Layers IDs': high_layers_ids,
            'Low Sat IDs': low_sat_ids
        }

    if not df_all.empty:
        grouped = df_all.groupby('config_label')
        for name, group in grouped:
            rows.append(get_count(group, name))
            
    if not df_best.empty:
        rows.append(get_count(df_best, 'Best Instances Per Configuration'))
        
    if not rows:
        return

    outlier_df = pd.DataFrame(rows)
    
    # Print exactly what we are putting into the PDF to the console
    print("\n" + "="*80)
    print("OUTLIER ANALYSIS (Matches PDF Table)")
    print("="*80)
    print(outlier_df.to_string(index=False))
    print("="*80 + "\n")
    
    
    fig, ax = plt.subplots(figsize=(12, len(outlier_df) * 0.5 + 2))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=outlier_df.values,
                     colLabels=outlier_df.columns,
                     loc='center',
                     cellLoc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)
    
    plt.title("Outlier Analysis\n(Layers > Median + 1SD and Sat % < Mean - 1SD)", y=0.98)
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

def plot_best_config_distribution(df_best, pdf):
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
    sns.barplot(x=config_counts.values, y=config_counts.index, palette="viridis")
    
    plt.title("Distribution of Best Configurations")
    plt.xlabel("Number of Instances")
    plt.ylabel("Configuration")
    plt.tight_layout()
    
    # Add counts to the end of bars
    for i, v in enumerate(config_counts.values):
        plt.text(v + 0.1, i, str(v), color='black', va='center')

    pdf.savefig()
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

    # Create Summary Table
    summary_table = create_summary_table(df_all, df_best)
    approx_ratio_table = create_approx_ratio_table(df_all, df_best)
    
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
    output_csv = f"{args.output}.csv"
    summary_table.to_csv(output_csv, index=False)
    print(f"Table saved to {output_csv}")

    # Generate PDF
    output_pdf = f"{args.output}.pdf"
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
        plt.title(f"Benchmark Summary: {os.path.basename(args.results_dir)}\n\n", y=0.98)
        
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
            # In matplotlib, doing two disjoint tables clearly on one plot is tricky.
            # Easiest way is to draw the second table below the first
            table2 = ax.table(cellText=approx_ratio_table.values,
                             colLabels=approx_ratio_table.columns,
                             loc='bottom',
                             bbox=[0.0, -0.6, 1.0, 0.4], # [x0, y0, width, height]
                             cellLoc='center')
            table2.auto_set_font_size(False)
            table2.set_fontsize(8)
            
            # Save the approximation ratio table to CSV as well
            output_approx_csv = f"{args.output}_approx_ratio.csv"
            approx_ratio_table.to_csv(output_approx_csv, index=False)
            print(f"Approximation Ratio Table saved to {output_approx_csv}")

        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Page 2: Distribution Plot
        plot_best_config_distribution(df_best, pdf)
        
        # Page 3: Outlier Counts
        plot_outlier_counts(df_all, df_best, pdf)
        
    print(f"PDF Report saved to {output_pdf}")

if __name__ == "__main__":
    main()
