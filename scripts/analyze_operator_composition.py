import pandas as pd
import json
import os
import glob
import argparse

def classify_operator(op_str):
    """
    Classify the Pauli string into:
    - 'Global QAOA Mixer' (sum of X's like X1+X2+...)
    - '1-Qubit' (single X, Y, or Z)
    - '2-Qubit' (e.g., X1Y2, Z3X4)
    - 'Other'
    """
    if '+' in op_str:
        if all(part.startswith('X') for part in op_str.split('+')):
            return 'Global QAOA Mixer'
        return 'Other'
        
    # Count letters X, Y, Z
    letters = [c for c in op_str if c in 'XYZ']
    
    if len(letters) == 1:
        return '1-Qubit'
    elif len(letters) == 2:
        return '2-Qubit'
    else:
        return 'Other'

def get_active_qubits(op_str):
    """
    Extracts all numeric qubit indices that are acted upon in the given Pauli string.
    Example: 'Z2X9' -> {2, 9}
    Example: 'X1+X2+X3' -> {1, 2, 3}
    """
    import re
    # Find all sequences of digits
    numbers = re.findall(r'\d+', op_str)
    return set(int(n) for n in numbers)

def load_operator_maps(results_dir):
    """
    Load all operator maps inside the given directory structure. 
    Keys are strings of the numeric IDs, values are strings of the Pauli representation.
    """
    map_files = glob.glob(os.path.join(results_dir, "**", "pool_operator_map_*.json"), recursive=True)
    op_map = {}
    
    for f_path in map_files:
        try:
            with open(f_path, 'r') as f:
                data = json.load(f)
                for k, v in data.items():
                    op_map[int(k)] = v
        except Exception as e:
            print(f"Warning: Could not parse {f_path}: {e}")
            
    # If no maps were found at all inside results_dir, look globally in temp/
    if not op_map:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(results_dir)))
        if "temp" in base_dir:
            temp_map_files = glob.glob(os.path.join(base_dir, "**", "pool_operator_map_*.json"), recursive=True)
            for f_path in temp_map_files:
                try:
                    with open(f_path, 'r') as f:
                        data = json.load(f)
                        for k, v in data.items():
                            op_map[int(k)] = v
                except:
                    pass
    
    return op_map

def get_operator_composition_data(results_dir):
    """
    Returns a DataFrame containing the operator composition counts per layer for all JSONs in the target directory.
    """
    op_map = load_operator_maps(results_dir)
    if len(op_map) == 0:
        print("Warning: No operator maps found. Will not be able to classify operators.")
    
    json_files = glob.glob(os.path.join(results_dir, "**", "*_all_*.json"), recursive=True)
    records = []
    
    for file_path in json_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            # Figure out total qubits from filename if possible, default to 10
            import re
            n_match = re.search(r'_N(\d+)_', file_path)
            total_qubits = int(n_match.group(1)) if n_match else 10
            
            for res in data.get('results', []):
                method = res.get('method', 'unknown').lower()
                gamma = res.get('initial_gamma', None)
                selected_indices = res.get('selected_indices', [])
                
                # Iterate over ALL layers performed
                for layer_idx in range(len(selected_indices)):
                    layer_ops = selected_indices[layer_idx]
                    
                    mixer_count = 0
                    q1_count = 0
                    q2_count = 0
                    active_qubits_this_layer = set()
                    sum_operator_qubit_lengths = 0
                    
                    for op_id in layer_ops:
                        op_str = op_map.get(op_id, "")
                        if not op_str:
                            continue
                            
                        cls = classify_operator(op_str)
                        if cls == 'Global QAOA Mixer':
                            mixer_count += 1
                        elif cls == '1-Qubit':
                            q1_count += 1
                        elif cls == '2-Qubit':
                            q2_count += 1
                            
                        op_qubits = get_active_qubits(op_str)
                        active_qubits_this_layer.update(op_qubits)
                        sum_operator_qubit_lengths += len(op_qubits)
                        
                    is_disjoint = (sum_operator_qubit_lengths == len(active_qubits_this_layer))
                            
                    records.append({
                        'method': method,
                        'gamma': gamma,
                        'layer': layer_idx + 1,
                        'global_qaoa_mixers': mixer_count,
                        '1_qubit_ops': q1_count,
                        '2_qubit_ops': q2_count,
                        'total_ops': len(layer_ops),
                        'layer_density': len(active_qubits_this_layer) / total_qubits,
                        'is_disjoint': is_disjoint
                    })
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            
    return pd.DataFrame(records)


            
def main():
    parser = argparse.ArgumentParser(description="Analyze composition of selected operators per layer")
    parser.add_argument("results_dir", help="Directory containing benchmark JSONs")
    args = parser.parse_args()
    
    print(f"Loading and classifying operator maps from {args.results_dir}...")
    df = get_operator_composition_data(args.results_dir)
            
    if df.empty:
        print("No valid records found.")
        return
        
    # Calculate averages
    summary = df.groupby(['gamma', 'layer', 'method']).agg({
        'global_qaoa_mixers': 'mean',
        '1_qubit_ops': 'mean',
        '2_qubit_ops': 'mean',
        'total_ops': 'mean',
        'layer_density': 'mean',
        'is_disjoint': 'all'
    }).round(3).reset_index()
    
    output_csv = os.path.join(args.results_dir, "operator_composition_summary.csv")
    summary.to_csv(output_csv, index=False)
    print(f"\nOperator Composition CSV successfully saved to {output_csv}\n")
    
    # Print console preview
    print(summary.to_string(index=False))

if __name__ == "__main__":
    main()
