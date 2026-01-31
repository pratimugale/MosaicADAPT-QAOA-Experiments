import argparse
import json
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import seaborn as sns
import os
from datetime import datetime

def load_data(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    metadata = {}
    if isinstance(data, dict) and 'results' in data:
        # New format
        df = pd.DataFrame(data['results'])
        metadata['config'] = data.get('config')
        metadata['threads'] = data.get('threads')
    else:
        # Old format (list of records)
        df = pd.DataFrame(data)
        # Try to infer threads from first row if it exists
        if 'threads' in df.columns:
             metadata['threads'] = df['threads'].iloc[0]
             
    return df, metadata

def get_longest_instances_per_n(df):
    """Finds the longest running instance for each N."""
    # Group by N, find index of max time
    idx = df.groupby('n_vars')['time_tetris'].idxmax()
    subset = df.loc[idx].copy()
    
    # Select and rename columns for the table
    # Check if 'instance_id' or 'instance_idx' exists
    id_col = 'instance_idx' if 'instance_idx' in subset.columns else 'instance_id'
    
    subset = subset[['n_vars', id_col, 'time_tetris', 'stop_reason']]
    subset['time_tetris'] = subset['time_tetris'].round(2)
    # Wrap headers: print each word in a new line
    subset.columns = ['N', 'Instance\nID', 'Time\n(s)', 'Stop\nReason']
    subset = subset.sort_values('N')
    return subset

def get_stats_dataframe(df):
    """Calculate summary stats and return as a nice DataFrame for the report."""
    # 1. Execution Time
    time_stats = df.groupby('n_vars')['time_tetris'].agg(['mean', 'median', 'max']).round(2)
    time_stats.columns = ['Time\nMean\n(s)', 'Time\nMedian\n(s)', 'Time\nMax\n(s)']
    
    # 2. Satisfaction
    sat_stats = df.groupby('n_vars')['satisfaction_tetris_percent'].agg(['mean', 'median', 'min']).round(2)
    sat_stats.columns = ['Sat\nMean\n%', 'Sat\nMedian\n%', 'Sat\nMin\n%']

    # 3. Layers
    layer_stats = df.groupby('n_vars')['layers'].agg(['mean', 'median', 'max']).round(1)
    layer_stats.columns = ['Layers\nMean', 'Layers\nMedian', 'Layers\nMax']
    
    # Combine
    combined = pd.concat([time_stats, sat_stats, layer_stats], axis=1)
    return combined

def get_longest_instance_info(df):
    """Finds the instance that took the longest time."""
    longest = df.loc[df['time_tetris'].idxmax()]
    return {
        'id': longest.get('instance_id', 'N/A'),
        'n_vars': longest['n_vars'],
        'time': longest['time_tetris'],
        'reason': longest.get('stop_reason', 'N/A')
    }

def plot_stopping_reasons(df):
    """Creates a stacked bar chart of stopping reasons."""
    # Count reasons per n_vars
    counts = df.groupby(['n_vars', 'stop_reason']).size().unstack(fill_value=0)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    counts.plot(kind='bar', stacked=True, ax=ax, colormap='viridis')
    
    ax.set_title('Distribution of Stopping Reasons vs Qubits')
    ax.set_xlabel('Number of Variables (N)')
    ax.set_ylabel('Count')
    ax.legend(title='Stopping Reason', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, axis='y', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_combined_scaling_figure(df):
    """Creates and returns the matplotlib figure for the combined plots."""
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
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
    return fig

def save_stats_csv(df, output_dir):
    # Reuse the stats logic
    stats = get_stats_dataframe(df)
    report = stats.transpose()
    output_path = os.path.join(output_dir, 'scaling_stats_summary.csv')
    report.to_csv(output_path)
    print(f"Saved stats report to {output_path}")

def create_pdf_report(df, output_dir, json_filename, metadata=None):
    """Generates a multipage PDF report."""
    output_path = os.path.join(output_dir, 'scaling_report.pdf')
    
    if metadata is None:
        metadata = {}

    with PdfPages(output_path) as pdf:
        # --- Page 1: Configuration & Metadata ---
        fig_config = plt.figure(figsize=(11.69, 8.27))
        fig_config.suptitle(f"Benchmark Configuration: {json_filename}", fontsize=16, y=0.95)
        
        txt_conf = f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        txt_conf += f"Total Records: {len(df)}\n"
        
        # Display Threads
        if 'threads' in metadata and metadata['threads']:
            txt_conf += f"Threads: {metadata['threads']}\n"
            
        txt_conf += "\nConfiguration Parameters:\n"
        txt_conf += "-" * 40 + "\n"
        
        config = metadata.get('config')
        if config:
            for key, value in config.items():
                txt_conf += f"{key}: {value}\n"
        else:
            txt_conf += "No configuration data found.\n"
            
        plt.figtext(0.1, 0.85, txt_conf, fontsize=12, ha='left', va='top', family='monospace')
        
        pdf.savefig(fig_config)
        plt.close(fig_config)

        # --- Page 2: Summary Tables ---
        fig_stats = plt.figure(figsize=(11.69, 8.27))
        fig_stats.suptitle("Benchmark Summary Statistics", fontsize=16, y=0.95)
        
        txt = f"Variables (N): {sorted(df['n_vars'].unique().tolist())}\n"
        plt.figtext(0.1, 0.85, txt, fontsize=12, ha='left', va='top')

        # Create Summary Table
        stats_df = get_stats_dataframe(df)
        stats_df = stats_df.reset_index().rename(columns={'n_vars': 'N'})
        # Cast to int then string to avoid decimal formatting
        stats_df['N'] = stats_df['N'].astype(int).astype(str)
        
        ax_table = plt.subplot(111)
        ax_table.axis('off')
        
        # Table position
        table = plt.table(cellText=stats_df.values, colLabels=stats_df.columns, 
                          loc='center', cellLoc='center', bbox=[0.1, 0.4, 0.8, 0.4])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        # Bold headers
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_text_props(weight='bold')
                cell.set_facecolor('#eeeeee')
        
        pdf.savefig(fig_stats)
        plt.close(fig_stats)

        # --- Page 3: Visualizations (Boxplots) ---
        fig_plots = plot_combined_scaling_figure(df)
        fig_plots.suptitle("Performance Scaling (Time, Satisfaction, Layers)", fontsize=14, y=0.98)
        pdf.savefig(fig_plots)
        plt.close(fig_plots)

        # --- Page 4: Stopping Reasons ---
        if 'stop_reason' in df.columns:
            fig_reasons = plot_stopping_reasons(df)
            pdf.savefig(fig_reasons)
            plt.close(fig_reasons)
            
        # --- Page 5: Stratified Longest Instances ---
        fig_strat = plt.figure(figsize=(11.69, 8.27))
        fig_strat.suptitle("Longest Running Instances per Qubit Count", fontsize=14, y=0.95)
        
        strat_df = get_longest_instances_per_n(df)
        
        ax_strat = plt.subplot(111)
        ax_strat.axis('off')
        
        # Create table
        table_strat = plt.table(cellText=strat_df.values, colLabels=strat_df.columns,
                                loc='upper center', cellLoc='center', bbox=[0.1, 0.6, 0.8, 0.3])
        table_strat.auto_set_font_size(False)
        table_strat.set_fontsize(10)
        table_strat.scale(1.2, 1.5) # Increase height for better visibility
        
        # Bold headers and styling
        for (row, col), cell in table_strat.get_celld().items():
            if row == 0:
                cell.set_text_props(weight='bold')
                cell.set_facecolor('#eeeeee')
        
        pdf.savefig(fig_strat)
        plt.close(fig_strat)
        
    print(f"Saved PDF report to {output_path}")

# ... [Keep helper functions unchanged] ...

def main():
    parser = argparse.ArgumentParser(description='Plot scaling benchmark results')
    parser.add_argument('json_file', type=str, help='Path to benchmark results JSON')
    parser.add_argument('--output', type=str, default='plots', help='Output directory for plots')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.output):
        os.makedirs(args.output)
        
    print(f"Loading data from {args.json_file}...")
    df, metadata = load_data(args.json_file)
    print(f"Loaded {len(df)} records.")
    
    # Print stats to console
    print(get_stats_dataframe(df))
    
    # 1. Save individual plot
    fig = plot_combined_scaling_figure(df)
    png_path = os.path.join(args.output, 'scaling_combined_boxplot.png')
    fig.savefig(png_path, dpi=300)
    print(f"Saved combined plot to {png_path}")
    
    # 2. Save stats CSV
    save_stats_csv(df, args.output)
    
    # 3. Create PDF Report
    json_filename = os.path.basename(args.json_file)
    create_pdf_report(df, args.output, json_filename, metadata)

if __name__ == "__main__":
    main()
