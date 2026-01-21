import argparse
import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

def load_data(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return pd.DataFrame(data)

def plot_combined_scaling(df, output_dir):
    # Print stats to stdout
    time_stats = df.groupby('n_vars')['time_tetris'].agg(['mean', 'median', 'max'])
    print("\n--- Execution Time Stats (s) ---")
    print(time_stats)
    
    sat_stats = df.groupby('n_vars')['satisfaction_tetris_percent'].agg(['mean', 'median', 'min'])
    print("\n--- Satisfaction Stats (%) ---")
    print(sat_stats)

    layer_stats = df.groupby('n_vars')['layers'].agg(['mean', 'median', 'max'])
    print("\n--- Layers Stats ---")
    print(layer_stats)

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # Plot 1: Time
    sns.boxplot(data=df, x='n_vars', y='time_tetris', palette='Set3', ax=axes[0])
    axes[0].set_title('Execution Time vs Qubits')
    axes[0].set_xlabel('Number of Variables (N)')
    axes[0].set_ylabel('Time (s)')
    axes[0].grid(True, axis='y', linestyle='-', alpha=0.5)
    
    # Plot 2: Satisfaction
    sns.boxplot(data=df, x='n_vars', y='satisfaction_tetris_percent', palette='Set2', ax=axes[1])
    axes[1].set_title('Satisfied Clauses % vs Qubits')
    axes[1].set_xlabel('Number of Variables (N)')
    axes[1].set_ylabel('Percent Satisfied')
    axes[1].grid(True, axis='y', linestyle='-', alpha=0.5)

    # Plot 3: Layers
    sns.boxplot(data=df, x='n_vars', y='layers', palette='Pastel1', ax=axes[2])
    axes[2].set_title('ADAPT Layers vs Qubits')
    axes[2].set_xlabel('Number of Variables (N)')
    axes[2].set_ylabel('Number of Layers')
    axes[2].grid(True, axis='y', linestyle='-', alpha=0.5)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'scaling_combined_boxplot.png')
    plt.savefig(output_path, dpi=300)
    print(f"Saved combined plot to {output_path}")
    plt.close()

def save_stats_csv(df, output_dir):
    # Group by N and calculate means
    means = df.groupby('n_vars')[['time_tetris', 'satisfaction_tetris_percent', 'layers']].mean()
    
    # Rename for clarity
    means.columns = ['Average Time (s)', 'Average Satisfaction', 'Average Layers']
    
    # Transpose so N are columns, Metrics are rows
    report = means.transpose()
    
    # Save
    output_path = os.path.join(output_dir, 'scaling_stats_summary.csv')
    report.to_csv(output_path)
    print(f"Saved stats report to {output_path}")
    print("\n--- Summary CSV Content ---")
    print(report)

def main():
    parser = argparse.ArgumentParser(description='Plot scaling benchmark results')
    parser.add_argument('json_file', type=str, help='Path to benchmark results JSON')
    parser.add_argument('--output', type=str, default='plots', help='Output directory for plots')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.output):
        os.makedirs(args.output)
        
    print(f"Loading data from {args.json_file}...")
    df = load_data(args.json_file)
    print(f"Loaded {len(df)} records.")
    
    plot_combined_scaling(df, args.output)
    save_stats_csv(df, args.output)

if __name__ == "__main__":
    main()
