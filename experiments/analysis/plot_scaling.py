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
    
def clean_dataframe(df):
    """Post-process dataframe to clean up fields."""
    # Replace empty stop reasons with a placeholder
    if 'stop_reason' in df.columns:
        df['stop_reason'] = df['stop_reason'].fillna('OptimizerMaxHeuristic')
        df['stop_reason'] = df['stop_reason'].replace('', 'OptimizerMaxHeuristic')
        
    # Calculate Approximation Ratio if data available
    if 'satisfaction_tetris_percent' in df.columns and 'satisfaction_rc2_percent' in df.columns:
        df['approximation_ratio'] = df.apply(
            lambda row: row['satisfaction_tetris_percent'] / row['satisfaction_rc2_percent'] 
            if row['satisfaction_rc2_percent'] > 1e-9 else 0.0, axis=1
        )
    return df

def get_convergence_data(df):
    """Explodes the adaptation_energies column into a long-format dataframe for plotting."""
    if 'adaptation_energies' not in df.columns or 'energy_floor_gurobi' not in df.columns:
        return pd.DataFrame()
    
    records = []
    # Iterate over each instance result
    for _, row in df.iterrows():
        energies = row['adaptation_energies']
        floor = row.get('energy_floor_gurobi')
        
        # Avoid division by zero if floor is 0 (though unlikely for MaxSAT)
        if floor == 0:
            floor = 1e-9

        if isinstance(energies, list) and len(energies) > 0 and floor is not None:
            for params_idx, energy in enumerate(energies):
                # Calculate Residual Energy: E_curr - E_floor
                # This works even if floor is 0.0
                residual = energy - floor
                
                records.append({
                    'n_vars': row['n_vars'],
                    'instance_idx': row.get('instance_idx', 0),
                    'layer': params_idx + 1,
                    'residual': residual
                })
    
    return pd.DataFrame(records)

def plot_convergence(df):
    """Creates a line plot of Residual Energy vs Layers with shaded std dev."""
    conv_df = get_convergence_data(df)
    
    if conv_df.empty:
        return None

    # Get unique N values
    n_values = sorted(conv_df['n_vars'].unique())
    num_plots = len(n_values)
    
    # Create subplots (one for each N)
    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 6), squeeze=False)
    axes = axes.flatten()
    
    for i, n in enumerate(n_values):
        ax = axes[i]
        subset = conv_df[conv_df['n_vars'] == n]
        
        # Plot Mean + Std Dev Shadow
        sns.lineplot(data=subset, x='layer', y='residual', errorbar='sd', ax=ax, marker='o')
        
        # Add optimality line (0.0)
        ax.axhline(0.0, color='red', linestyle='--', label='Optimal (0.0)', alpha=0.7)
        
        ax.set_title(f"Convergence (N={n})")
        ax.set_xlabel("Adaptation Layer")
        ax.set_ylabel("Residual Energy (E - Floor)")
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.legend()
        
    plt.tight_layout()
    return fig

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
    
    objs = [time_stats, sat_stats, layer_stats]

    # 4. Approximation Ratio (Optional)
    if 'approximation_ratio' in df.columns:
        approx_stats = df.groupby('n_vars')['approximation_ratio'].agg(['mean', 'min']).round(3)
        approx_stats.columns = ['Approx\nMean', 'Approx\nMin']
        objs.append(approx_stats)
    
    # Combine
    combined = pd.concat(objs, axis=1)
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
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
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
    
    # Add Layer Limit Line if available
    if 'layer_limit' in df.columns:
        # Calculate mean limit (should be constant per N)
        limit_data = df.groupby('n_vars')['layer_limit'].mean().reset_index()
        sns.lineplot(data=limit_data, x='n_vars', y='layer_limit', ax=axes[2], 
                     color='red', linestyle='--', label='Limit (2*N)', marker='o')
        axes[2].legend()
        
    # Plot 4: Approximation Ratio
    if 'approximation_ratio' in df.columns:
        sns.boxplot(data=df, x='n_vars', y='approximation_ratio', palette='Set1', ax=axes[3])
        axes[3].set_title('Approximation Ratio (Tetris / RC2)')
        axes[3].set_xlabel('Number of Variables (N)')
        axes[3].set_ylabel('Ratio')
        axes[3].grid(True, axis='y', linestyle='-', alpha=0.5)
        # Add optimality line
        axes[3].axhline(1.0, color='red', linestyle='--', label='Optimal (1.0)')
        axes[3].legend()
    else:
        axes[3].text(0.5, 0.5, 'Approximation Data Not Available', 
                     ha='center', va='center', fontsize=12, color='gray')
        axes[3].axis('off')
    
    plt.tight_layout()
    return fig

def save_stats_csv(df, output_dir):
    # Reuse the stats logic
    stats = get_stats_dataframe(df)
    report = stats.transpose()
    output_path = os.path.join(output_dir, 'scaling_stats_summary.csv')
    report.to_csv(output_path)
    print(f"Saved stats report to {output_path}")

def create_pdf_report(df, output_dir, json_filename, metadata=None, fig_combined=None):
    """Generates a multipage PDF report."""
    output_path = os.path.join(output_dir, 'scaling_report.pdf')
    
    if metadata is None:
        metadata = {}

    with PdfPages(output_path) as pdf:
        # --- Page 1: Configuration & Metadata ---
        fig_config = plt.figure(figsize=(11.69, 8.27))
        fig_config.suptitle(f"Benchmark Configuration: {json_filename}", fontsize=16, y=0.95)
        
        txt_conf = f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        
        # Breakdown instances per N
        counts = df['n_vars'].value_counts().sort_index()
        count_str = ", ".join([f"N={n}: {c}" for n, c in counts.items()])
        txt_conf += f"Instances per Qubit Count (N): {count_str}\n"
        
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
        if fig_combined:
            fig_plots = fig_combined
        else:
            fig_plots = plot_combined_scaling_figure(df)
            
        fig_plots.suptitle("", fontsize=14, y=0.98)
        pdf.savefig(fig_plots)
        # Only close if we created it locally, or let caller handle it? 
        # Actually safer to let PdfPages verify it's saved. 
        # Since we use it in main, we should probably not close it here if passed in, 
        # BUT main doesn't use it after this call. So we can close it.
        plt.close(fig_plots)

        if 'stop_reason' in df.columns:
            fig_reasons = plot_stopping_reasons(df)
            pdf.savefig(fig_reasons)
            plt.close(fig_reasons)

        # --- Page 5: Convergence Trajectories ---
        fig_conv = plot_convergence(df)
        if fig_conv:
            fig_conv.suptitle("Energy Convergence vs Adaptation Layers (Mean ± SD)", fontsize=14, y=0.98)
            pdf.savefig(fig_conv)
            plt.close(fig_conv)
            

            
        # --- Page 6: Stratified Longest Instances ---
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
    df = clean_dataframe(df)
    print(f"Loaded {len(df)} records.")
    
    # Print stats to console
    print(get_stats_dataframe(df))
    
    # 1. Generate Combined Figure Once
    fig_combined = plot_combined_scaling_figure(df)
    
    # 2. Save PNG
    png_path = os.path.join(args.output, 'scaling_combined_boxplot.png')
    fig_combined.savefig(png_path, dpi=300)
    print(f"Saved combined plot to {png_path}")
    
    # 3. Save Stats CSV
    save_stats_csv(df, args.output)
    
    # 4. Create PDF Report (Passing the existing figure)
    json_filename = os.path.basename(args.json_file)
    create_pdf_report(df, args.output, json_filename, metadata, fig_combined=fig_combined)

if __name__ == "__main__":
    main()
