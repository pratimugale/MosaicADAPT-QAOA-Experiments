import json
import sys

def find_increase(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    results = data.get('results', [])
    found = False
    
    for res in results:
        energies = res.get('adaptation_energies', [])
        n_vars = res.get('n_vars')
        config = res.get('config_label')
        instance = res.get('instance_idx')
        
        # Check for increase
        for i in range(len(energies) - 1):
            if energies[i+1] > energies[i] + 1e-9: # tolerance for float
                print(f"FOUND INCREASE:")
                print(f"  N={n_vars}, Config='{config}', Instance={instance}")
                print(f"  Layer {i+1} -> {i+2}")
                print(f"  Energy: {energies[i]} -> {energies[i+1]}")
                print(f"  Diff: {energies[i+1] - energies[i]}")
                print("-" * 30)
                found = True
                
    if not found:
        print("No energy increases found.")

if __name__ == "__main__":
    find_increase(sys.argv[1])
