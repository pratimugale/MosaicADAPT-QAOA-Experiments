import os
import sys
import random
import argparse
import json
import hashlib

# Locate paths relative to this script
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))

# Add project root to sys.path to import from src
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from src.dataset.satqubolib_max3sat import generate_max3sat_instance
    from src.dataset.random_max3sat import generate_random_max3sat_instance
except ImportError as e:
    print(f"Error: Could not import generation logic: {e}")
    sys.exit(1)


def get_canonical_string_and_hash(cnf_path):
    clauses = []
    with open(cnf_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('c') or line.startswith('p') or line.startswith('%'):
                continue
            
            tokens = line.split()
            lits = []
            for t in tokens:
                if t == '%':
                    break
                val = int(t)
                if val == 0:
                    break
                lits.append(val)
                
            if lits:
                clauses.append(lits)
                
    def literal_sort_key(lit):
        var = abs(lit)
        return (var, 0 if lit > 0 else 1)
        
    sorted_clauses = []
    for c in clauses:
        sorted_clauses.append(sorted(c, key=literal_sort_key))
        
    def clause_sort_key(c):
        return tuple(literal_sort_key(lit) for lit in c)
        
    sorted_clauses.sort(key=clause_sort_key)
    
    clause_strs = []
    for c in sorted_clauses:
        lit_strs = []
        for lit in c:
            if lit > 0:
                lit_strs.append(f"x{lit}")
            else:
                lit_strs.append(f"~x{-lit}")
        clause_strs.append("(" + " v ".join(lit_strs) + ")")
        
    canonical_str = " ^ ".join(clause_strs)
    h = hashlib.sha256(canonical_str.encode('utf-8')).hexdigest()
    
    return canonical_str, h


class AntiLeakageRegistry:
    def __init__(self, map_path):
        self.map_path = map_path
        self.registry = []
        self.seen_hashes = set()
        
        if os.path.exists(self.map_path):
            try:
                with open(self.map_path, 'r') as f:
                    self.registry = json.load(f)
                    for item in self.registry:
                        if 'hash' in item:
                            self.seen_hashes.add(item['hash'])
            except json.JSONDecodeError:
                pass
                
    def is_duplicate(self, h):
        return h in self.seen_hashes
        
    def add_instance(self, instance_dict):
        self.registry.append(instance_dict)
        self.seen_hashes.add(instance_dict['hash'])
        self.save()
        
    def save(self):
        os.makedirs(os.path.dirname(self.map_path), exist_ok=True)
        with open(self.map_path, 'w') as f:
            json.dump(self.registry, f, indent=2)


def generate_dataset(n_vars, num_instances, type_name, seed):
    if type_name == "balanced":
        base_ratio = 3.6
    elif type_name == "triangle":
        base_ratio = 4.3
    elif type_name == "random":
        base_ratio = 4.3
    else:
        print(f"Error: Unknown instance type {type_name}")
        sys.exit(1)

    print(f"Generating {num_instances} {type_name} instances for N={n_vars} (base ratio {base_ratio})...")

    registry_path = os.path.join(project_root, "dataset", "satqubolib", "canonical_formula_map.json")
    registry = AntiLeakageRegistry(registry_path)

    for i in range(num_instances):
        seed_offset = 0
        while True:
            # Seed RNG deterministically for this exact attempt to preserve reproducibility
            attempt_seed = seed + i + seed_offset
            random.seed(attempt_seed)
            
            # Randomize ratio +/- 20%
            variation = (random.random() * 0.4) - 0.2
            instance_ratio = base_ratio * (1.0 + variation)
            
            try:
                if type_name == "random":
                    path, used_seed = generate_random_max3sat_instance(
                        n_vars, 
                        instance_ratio, 
                        seed=seed + seed_offset, 
                        instance_idx=i, 
                        num_instances=1 
                    )
                else:
                    path, used_seed = generate_max3sat_instance(
                        type_name, 
                        n_vars, 
                        instance_ratio, 
                        seed=seed + seed_offset, 
                        instance_idx=i, 
                        num_instances=1 
                    )
                
                canonical_str, h = get_canonical_string_and_hash(path)
                
                if registry.is_duplicate(h):
                    print(f"⚠️  WARNING: Duplicate instance detected for i={i} (Hash Collision: {h[:8]}...). Regenerating...")
                    if os.path.exists(path):
                        os.remove(path)
                    seed_offset += 1000
                else:
                    filename = os.path.basename(path)
                    
                    registry.add_instance({
                        "instance_idx": i,
                        "formula": canonical_str,
                        "type": type_name,
                        "filename": filename,
                        "hash": h
                    })
                    print(f"Instance {i+1}/{num_instances} saved to: {path} (Hash: {h[:8]})")
                    break
                    
            except Exception as e:
                print(f"Error generating instance {i}: {e}")
                break

    print("Generation complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Max-3SAT problem instances (Balanced, No Triangle, or Random).")
    parser.add_argument("n_vars", type=int, help="Number of variables per instance")
    parser.add_argument("num_instances", type=int, help="Number of instances to generate")
    parser.add_argument("--type", type=str, choices=["balanced", "triangle", "random"], default="balanced", help="Type of instances to generate")
    parser.add_argument("--seed", type=int, default=2000, help="Random seed")
    
    args = parser.parse_args()
    
    generate_dataset(args.n_vars, args.num_instances, args.type, args.seed)
