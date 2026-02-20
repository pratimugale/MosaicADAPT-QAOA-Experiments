
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

def create_summary_table(df_all, df_best):
    """
    Generates a summary DataFrame with Mean, Median, Max for:
    - Time (adapt_time)
    - Layers (layers)
    - Satisfaction (satisfaction_percent)
    
    Rows: Each configuration + "Best Instances Per Configuration"
    """
    stats_cols = ['adapt_time', 'layers', 'satisfaction_percent']
    
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
            summary_rows.append(row)

    # 2. Process "Best" Configuration (The Hybrid/Oracle result)
    if not df_best.empty:
        row = {'Configuration': 'Best Instances Per Configuration'}
        for col in stats_cols:
            row[f'{col} (Mean)'] = df_best[col].mean()
            row[f'{col} (Median)'] = df_best[col].median()
            row[f'{col} (Max)'] = df_best[col].max()
        summary_rows.append(row)
        
    summary_df = pd.DataFrame(summary_rows)
    
    # Reorder columns for clarity
    # Config | Sat (Mean, Med, Max) | Layers (Mean, Med, Max) | Time (Mean, Med, Max)
    ordered_cols = ['Configuration']
    labels = ['Satisfaction', 'Layers', 'Time']
    orig_cols = ['satisfaction_percent', 'layers', 'adapt_time']
    
    final_cols = ['Configuration']
    for label, orig in zip(labels, orig_cols):
        final_cols.append(f'{orig} (Mean)')
        final_cols.append(f'{orig} (Median)')
        final_cols.append(f'{orig} (Max)')
        
    # Filter only existing columns
    final_cols = [c for c in final_cols if c in summary_df.columns]
    
    return summary_df[final_cols]

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
    
    print(f"Loaded {len(df_all)} records for individual configs.")
    print(f"Loaded {len(df_best)} records for best instances.")

    if df_all.empty and df_best.empty:
        print("No data found. Exiting.")
        return

    # Create Summary Table
    summary_table = create_summary_table(df_all, df_best)
    
    # Rename columns for display
    rename_map = {
        'adapt_time (Mean)': 'Time Mean (s)',
        'adapt_time (Median)': 'Time Med (s)',
        'adapt_time (Max)': 'Time Max (s)',
        'layers (Mean)': 'Layers Mean',
        'layers (Median)': 'Layers Med',
        'layers (Max)': 'Layers Max',
        'satisfaction_percent (Mean)': 'Sat % Mean',
        'satisfaction_percent (Median)': 'Sat % Med',
        'satisfaction_percent (Max)': 'Sat % Max'
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
        
        plt.title(f"Benchmark Summary: {os.path.basename(args.results_dir)}", y=0.98)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Page 2: Distribution Plot
        plot_best_config_distribution(df_best, pdf)
        
    print(f"PDF Report saved to {output_pdf}")

if __name__ == "__main__":
    main()
