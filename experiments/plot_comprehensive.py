import argparse
import json
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import seaborn as sns
import os
from datetime import datetime
import numpy as np

def load_data(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    metadata = {}
    if isinstance(data, dict) and 'results' in data:
        df = pd.DataFrame(data['results'])
        metadata['config'] = data.get('config')
        metadata['threads'] = data.get('threads')
        metadata['timestamp'] = data.get('timestamp')
    else:
        df = pd.DataFrame(data)
        
    return df, metadata

def process_traces(df):
    """
    Extracts residual energy traces (Energy - Floor).
    Returns a long-format DataFrame for plotting.
    """
    if 'adaptation_energies' not in df.columns or 'energy_floor' not in df.columns:
        return pd.DataFrame()
        
    records = []
    
    # First pass: Collect all traces and find max length
    all_traces = []
    
    # max_layers = 2 * N_max (approx). Let's just find the max length encountered.
    max_len_global = 0
    
    for _, row in df.iterrows():
        energies = row.get('adaptation_energies', [])
        floor = row.get('energy_floor')
        n_vars = row['n_vars']
        config = row.get('config_label', 'Unknown')
        
        if floor is None: continue
        
        if isinstance(energies, list) and len(energies) > 0:
            trace = []
            for energy in energies:
                trace.append(energy - floor)
            
            all_traces.append({
                'n_vars': n_vars,
                'config': config,
                'trace': trace
            })
            max_len_global = max(max_len_global, len(trace))

    # Second pass: Forward fill and flatten
    # Actually, we should forward fill PER CONFIG/N group, or just up to the max observed layer?
    # Usually, we plot up to the max layer any instance reached.
    
    for item in all_traces:
        trace = item['trace']
        n_vars = item['n_vars']
        config = item['config']
        
        # Forward fill this trace to max_len_global
        # This treats "finished early" as "staying at final value"
        current_len = len(trace)
        final_val = trace[-1]
        
        # Create full length trace
        full_trace = trace + [final_val] * (max_len_global - current_len)
        
        for layer_idx, residual in enumerate(full_trace):
             records.append({
                'n_vars': n_vars,
                'config': config,
                'layer': layer_idx + 1,
                'residual_energy': residual
            })
        
    return pd.DataFrame(records)

def process_satisfaction_traces(df):
    """
    Extracts satisfaction traces.
    Returns a long-format DataFrame for plotting.
    """
    if 'adaptation_clause_satisfaction_percent_trace' not in df.columns:
        return pd.DataFrame()
        
    records = []
    
    # First pass: Collect all traces and find max length
    all_traces = []
    
    max_len_global = 0
    
    for _, row in df.iterrows():
        traces = row.get('adaptation_clause_satisfaction_percent_trace', [])
        n_vars = row['n_vars']
        config = row.get('config_label', 'Unknown')
        
        if isinstance(traces, list) and len(traces) > 0:
            all_traces.append({
                'n_vars': n_vars,
                'config': config,
                'trace': traces
            })
            max_len_global = max(max_len_global, len(traces))

    # Second pass: Forward fill and flatten
    for item in all_traces:
        trace = item['trace']
        n_vars = item['n_vars']
        config = item['config']
        
        # Forward fill this trace to max_len_global
        current_len = len(trace)
        final_val = trace[-1]
        
        # Create full length trace
        full_trace = trace + [final_val] * (max_len_global - current_len)
        
        for layer_idx, sat in enumerate(full_trace):
             records.append({
                'n_vars': n_vars,
                'config': config,
                'layer': layer_idx + 1,
                'satisfaction': sat * 100.0
            })
        
    return pd.DataFrame(records)

def plot_convergence_by_n(conv_df, n_vars, ax):
    """Plots spread of Residual Energy vs layers for a specific N."""
    if conv_df.empty:
        return

    subset = conv_df[conv_df['n_vars'] == n_vars]
    
    if subset.empty:
        return
    
    # Plot Mean + Std Deviation
    sns.lineplot(
        data=subset, 
        x='layer', 
        y='residual_energy', 
        hue='config', 
        style='config',
        estimator='mean',
        errorbar=None, # Show only the mean
        ax=ax,
        linewidth=2,
        palette='tab10'
    )
    
    ax.set_title(f"Residual Energy Convergence (N={n_vars})")
    ax.set_xlabel("Adaptation Layer")
    ax.set_ylabel("Residual Energy (E - E_opt)")
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_yscale('log') # Residuals often span orders of magnitude
    
    # Legend handling
    ax.legend(title='Configuration', bbox_to_anchor=(1.05, 1), loc='upper left')

def plot_satisfaction_convergence_by_n(sat_df, n_vars, ax):
    """Plots spread of Satisfaction vs layers for a specific N."""
    if sat_df.empty:
        return

    subset = sat_df[sat_df['n_vars'] == n_vars]
    
    if subset.empty:
        return
    
    # Plot Mean (Log Scale)
    sns.lineplot(
        data=subset, 
        x='layer', 
        y='satisfaction', 
        hue='config', 
        style='config',
        estimator='mean',
        errorbar=None, 
        ax=ax,
        linewidth=2,
        palette='tab10'
    )
    
    ax.set_title(f"Clause Satisfaction (Log Scale, N={n_vars})")
    ax.set_xlabel("Adaptation Layer")
    ax.set_ylabel("Satisfied Clauses (%)")
    ax.grid(True, linestyle='--', alpha=0.3, which='both')
    ax.set_yscale('log')
    # Focus on the top part as requested
    ax.set_ylim(87.5, 100.5) 
    
    # Legend handling
    ax.legend(title='Configuration', bbox_to_anchor=(1.05, 1), loc='upper left')

def plot_final_residual_boxplot(df):
    """Boxplot of final residual energies."""
    if 'residual_final' not in df.columns:
        df['residual_final'] = df.apply(
            lambda row: row['energy'] - row['energy_floor'], axis=1
        )

    # DROP NaNs
    plot_df = df.dropna(subset=['residual_final', 'n_vars', 'config_label'])
    
    print(f"Plotting Boxplot with {len(plot_df)} records (original {len(df)})")
    if len(plot_df) == 0:
        print("WARNING: No valid data for boxplot!")
        return plt.figure()

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=plot_df, x='n_vars', y='residual_final', hue='config_label', ax=ax, palette='Set3')
    
    ax.set_title("Final Residual Energy Distribution")
    ax.set_xlabel("Number of Variables (N)")
    ax.set_ylabel("Final Residual (E - E_opt)")
    ax.grid(True, axis='y', linestyle='--', alpha=0.3)
    ax.set_yscale('log')
    ax.legend(title='Configuration', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    return fig

def create_report(df, metadata, output_dir, filename):
    output_path = os.path.join(output_dir, "comprehensive_report.pdf")
    
    # Process convergence data
    # Process convergence data
    conv_df = process_traces(df)
    sat_df = process_satisfaction_traces(df)
    
    with PdfPages(output_path) as pdf:
        # --- Page 1: Configuration ---
        fig_config = plt.figure(figsize=(11, 8))
        fig_config.suptitle(f"Benchmark Configuration\n{filename}", fontsize=16)
        
        # 1. General Info & Configurations
        txt_header = f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
        txt_header += f"Qubit Counts (N): {sorted(df['n_vars'].unique())}\n"
        txt_header += f"Threads: {metadata.get('threads', 'N/A')}\n\n"
        
        txt_header += "Configurations Benchmarked:\n"
        if 'config_label' in df.columns:
            configs = df['config_label'].unique()
            for c in configs:
                txt_header += f" - {c}\n"
        
        plt.figtext(0.1, 0.8, txt_header, fontsize=11, ha='left', va='top', family='monospace')

        # 2. Configuration Parameters
        txt_params = "Configuration Parameters:\n"
        txt_params += "-" * 40 + "\n"
        
        # safely get config dict
        run_config = metadata.get('config', {})
        if isinstance(run_config, dict):
            # Sort keys and exclude redundant/large fields
            sorted_keys = sorted(run_config.keys())
            for key in sorted_keys:
                if key == 'configurations': 
                    continue # Skip the large list, it's already summarized above
                
                val = run_config[key]
                txt_params += f"{key}: {val}\n"
        else:
            txt_params += "No configuration metadata found.\n"
            
        plt.figtext(0.1, 0.55, txt_params, fontsize=9, ha='left', va='top', family='monospace')
        pdf.savefig(fig_config)
        plt.close(fig_config)

        # --- Page 2: Detailed Statistics ---
        fig_stats = plt.figure(figsize=(11, 8))
        fig_stats.suptitle("Summary Statistics", fontsize=16)


        # 2. Aggregation
        # Group by N, Config
        # Metrics: layers, adapt_time
        stats = df.groupby(['n_vars', 'config_label']).agg({
            'layers': ['mean', 'median', 'max'],
            'adapt_time': ['mean', 'median', 'max']
        }).round(4)
        
        # Flatten MultiIndex columns
        stats.columns = ['_'.join(col).strip() for col in stats.columns.values]
        stats = stats.reset_index()
        
        # Rename for display
        display_cols = [
            'n_vars', 'config_label',
            'layers_mean', 'layers_median', 'layers_max',
            'adapt_time_mean', 'adapt_time_median', 'adapt_time_max'
        ]
        
        # Filter to available columns just in case
        valid_cols = [c for c in display_cols if c in stats.columns]
        table_data = stats[valid_cols]
        
        # Format column headers for the table
        headers = [
            "N", "Config", 
            "Layers\nMean", "Layers\nMed", "Layers\nMax",
            "Adapt Time\nMean", "Adapt Time\nMed", "Adapt Time\nMax"
        ]

        # Paginate the table
        rows_per_page = 20
        total_rows = len(table_data)
        num_pages = (total_rows + rows_per_page - 1) // rows_per_page
        
        for p in range(num_pages):
            fig_stats = plt.figure(figsize=(11, 8))
            if p == 0:
                fig_stats.suptitle("Summary Statistics", fontsize=16)
            else:
                fig_stats.suptitle("Summary Statistics (Cont.)", fontsize=16)

            ax_table = plt.subplot(111)
            ax_table.axis('off')
            
            start_idx = p * rows_per_page
            end_idx = min((p + 1) * rows_per_page, total_rows)
            page_data = table_data.iloc[start_idx:end_idx]
            cell_text = page_data.values.tolist()
            
            # Custom column widths: Small for N, Broad for Config, Equal for others
            # Total width approx 1.0 (relative to bbox width)
            # N=0.05, Config=0.25, Others=(0.7/9) ~0.077 each
            n_cols = len(headers)
            col_widths = [0.05, 0.25] + [0.115] * (n_cols - 2)

            table = plt.table(
                cellText=cell_text, 
                colLabels=headers,
                colWidths=col_widths, 
                loc='center', 
                bbox=[0.05, 0.1, 0.9, 0.8]
            )
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            
            pdf.savefig(fig_stats)
            plt.close(fig_stats)

        # --- Page 3+: Termination Reasons (One page per N) ---
        if 'stop_reason' in df.columns:
            df['clean_reason'] = df['stop_reason'].replace('', 'Unknown')
            unique_n = sorted(df['n_vars'].unique())
            
            for n_val in unique_n:
                subset = df[df['n_vars'] == n_val]
                if subset.empty: continue
                
                # Count frequency: Rows=Config, Cols=Reason
                reason_counts = pd.crosstab(subset['config_label'], subset['clean_reason'])
                
                fig_reasons = plt.figure(figsize=(11, 8))
                ax_reasons = plt.gca()
                
                # Plot stacked bar
                reason_counts.plot(kind='bar', stacked=True, ax=ax_reasons, colormap='viridis', width=0.6)
                
                ax_reasons.set_title(f"Stopping Reasons Distribution (N={n_val})", fontsize=16)
                ax_reasons.set_xlabel("Configuration", fontsize=12)
                ax_reasons.set_ylabel("Count", fontsize=12)
                ax_reasons.grid(axis='y', linestyle='--', alpha=0.3)
                plt.xticks(rotation=15, ha='right')
                
                # Legend outside
                plt.legend(title="Stopping Reason", bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.tight_layout()
                
                pdf.savefig(fig_reasons)
                plt.close(fig_reasons)
        
                # pdf.savefig(fig_reasons)
                # plt.close(fig_reasons)
        
        # --- Page 4: Longest Running Instances ---
        # Find max time instance per (N, Config)
        time_metric = 'adapt_time' if 'adapt_time' in df.columns else 'time'
        
        # Get indices of max time
        idx_max = df.groupby(['n_vars', 'config_label'])[time_metric].idxmax()
        longest_runs = df.loc[idx_max].sort_values(['n_vars', 'config_label'])
        
        # Select columns
        cols_to_show = ['n_vars', 'config_label', 'instance_idx', time_metric, 'stop_reason']
        table_data_longest = longest_runs[cols_to_show].copy()
        
        # Round time to 2 decimal places
        table_data_longest[time_metric] = table_data_longest[time_metric].round(2)
        
        # Rename columns
        col_map = {
            'n_vars': 'N',
            'config_label': 'Config',
            'instance_idx': 'Instance ID',
            time_metric: 'Time (s)',
            'stop_reason': 'Stop Reason'
        }
        table_data_longest = table_data_longest.rename(columns=col_map)
        
        # Create Table Page
        fig_longest = plt.figure(figsize=(11, 8))
        fig_longest.suptitle("Longest Running Instances per Configuration", fontsize=16)
        
        ax_long = plt.subplot(111)
        ax_long.axis('off')
        
        cell_text_long = table_data_longest.values.tolist()
        headers_long = table_data_longest.columns.tolist()
        
        # Adjust widths: Broad Config, others standard
        # N=0.05, Config=0.30, ID=0.1, Time=0.15, Reason=0.3
        col_widths_long = [0.05, 0.30, 0.1, 0.15, 0.3]
        
        table_long = plt.table(
            cellText=cell_text_long,
            colLabels=headers_long,
            colWidths=col_widths_long,
            loc='center',
            bbox=[0.05, 0.1, 0.9, 0.8]
        )
        table_long.auto_set_font_size(False)
        table_long.set_fontsize(9)
        
        pdf.savefig(fig_longest)
        plt.close(fig_longest)

        # --- Page 5: Final Boxplots ---
        fig_box = plot_final_residual_boxplot(df)
        pdf.savefig(fig_box)
        plt.close(fig_box)
        
        # --- Page 3+: Convergence Plots (One per N) ---
        n_values = sorted(df['n_vars'].unique())
        
        for n in n_values:
            fig, ax = plt.subplots(figsize=(10, 6))
            plot_convergence_by_n(conv_df, n, ax)
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        # --- Page 6+: Satisfaction Plots (One per N) ---
        if not sat_df.empty:
            for n in n_values:
                fig, ax = plt.subplots(figsize=(10, 6))
                plot_satisfaction_convergence_by_n(sat_df, n, ax)
                plt.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)
            
    print(f"Report saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Plot comprehensive benchmark results')
    parser.add_argument('json_file', type=str, help='Path to benchmark results JSON')
    parser.add_argument('--output', type=str, default='plots', help='Output directory for plots')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.output):
        os.makedirs(args.output)
        
    print(f"Loading {args.json_file}...")
    df, metadata = load_data(args.json_file)
    print(f"Loaded {len(df)} records.")
    
    create_report(df, metadata, args.output, os.path.basename(args.json_file))

if __name__ == "__main__":
    main()
