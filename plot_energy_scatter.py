import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

def plot_scatter():
    # Load data
    filepath = 'results/benchmark_comprehensive_2026-02-06-23-01-56.json'
    with open(filepath, 'r') as f:
        data = json.load(f)

    # Process data
    records = []
    
    target_configs = {
        'KaMIS_NewPool': 'KaMIS',
        'Greedy_NewPool': 'Greedy'
    }

    for result in data['results']:
        if result['n_vars'] != 15:
            continue

        config_label = result.get('config_label')
        if config_label not in target_configs:
            continue
        
        satisfaction = result.get('satisfaction_percent', 0.0)

        records.append({
            'seed': result['seed'],
            'n_vars': result['n_vars'],
            'method': target_configs[config_label],
            'satisfaction': satisfaction
        })

    df = pd.DataFrame(records)

    # Add a run_id to distinguish multiple runs for the same seed/method
    df['run_id'] = df.groupby(['seed', 'n_vars', 'method']).cumcount()

    # Pivot to align KaMIS and Greedy
    try:
        df_pivot = df.pivot(index=['seed', 'n_vars', 'run_id'], columns='method', values=['satisfaction']).reset_index()
    except ValueError as e:
        print(f"Error pivoting data: {e}")
        return

    # Flatten columns after pivot
    df_pivot.columns = ['_'.join(col).strip() if col[1] else col[0] for col in df_pivot.columns.values]
    
    # Drop rows where any method is missing
    df_pivot = df_pivot.dropna()
    print(f"Plotting {len(df_pivot)} paired data points (N=15).")

    # Rename columns for easier access
    df_pivot['Greedy'] = df_pivot['satisfaction_Greedy']
    df_pivot['KaMIS'] = df_pivot['satisfaction_KaMIS']

    # Calculate Win Rate (Higher satisfaction is better)
    wins = df_pivot[df_pivot['KaMIS'] > df_pivot['Greedy']]
    draws = df_pivot[df_pivot['KaMIS'] == df_pivot['Greedy']]
    
    print(f"\nAnalysis of Final Satisfaction ({len(df_pivot)} paired instances):")
    print(f"KaMIS > Greedy: {len(wins)} times")
    print(f"KaMIS == Greedy: {len(draws)} times")
    print(f"Win Rate (Strict): {len(wins) / len(df_pivot) * 100:.2f}%")

    # Plot
    plt.figure(figsize=(10, 8))
    
    # Get unique N values (should be just 15)
    n_values = sorted(df_pivot['n_vars'].unique())
    # Use a specific color for N=15
    color = 'gold' 

    for n in n_values:
        subset = df_pivot[df_pivot['n_vars'] == n]
        
        plt.scatter(subset['Greedy'], subset['KaMIS'], 
                    label=f'N={n}', color=color, alpha=0.6, edgecolors='w', s=60)

    # Diagonal line
    min_val = min(df_pivot['Greedy'].min(), df_pivot['KaMIS'].min())
    max_val = max(df_pivot['Greedy'].max(), df_pivot['KaMIS'].max())
    
    # Add some padding
    padding = (max_val - min_val) * 0.05 if max_val != min_val else 0.1
    # Ensure range covers 0-1 if relevant, but let's stick to data range
    
    plt.plot([min_val - padding, max_val + padding], [min_val - padding, max_val + padding], 
             'k--', alpha=0.5, label='y=x')

    plt.xlabel('Greedy Satisfaction Ratio', fontsize=12)
    plt.ylabel('KaMIS Satisfaction Ratio', fontsize=12)
    plt.title('Final Satisfaction Ratio: KaMIS vs Greedy (N=15)', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Ensure plots directory exists
    os.makedirs('plots', exist_ok=True)
    
    output_path = 'plots/satisfaction_scatter_n15_final.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    plot_scatter()
