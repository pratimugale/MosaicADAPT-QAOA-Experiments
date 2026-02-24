
import os
import sys
import random
import argparse

# Locate paths relative to this script
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))

# Add project root to sys.path to import from src
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from src.dataset.satqubolib_max3sat import generate_max3sat_instance
except ImportError as e:
    print(f"Error: Could not import generation logic: {e}")
    sys.exit(1)

def generate_dataset(n_vars, num_instances, type_name, seed):
    if type_name == "balanced":
        base_ratio = 3.6
    elif type_name == "triangle":
        base_ratio = 4.3
    else:
        print(f"Error: Unknown instance type {type_name}")
        sys.exit(1)

    # Seed RNG
    random.seed(seed)

    print(f"Generating {num_instances} {type_name} instances for N={n_vars} (base ratio {base_ratio})...")

    for i in range(num_instances):
        # Randomize ratio +/- 20%
        # random() gives [0, 1). We want [-0.2, 0.2].
        variation = (random.random() * 0.4) - 0.2
        instance_ratio = base_ratio * (1.0 + variation)
        
        # Call generation function directly
        try:
            path, used_seed = generate_max3sat_instance(
                type_name, 
                n_vars, 
                instance_ratio, 
                seed=seed, 
                instance_idx=i, 
                num_instances=1 # Always treat as 1 to get standard naming from ratio
            )
            print(f"Instance {i+1}/{num_instances} saved to: {path}")
        except Exception as e:
            print(f"Error generating instance {i}: {e}")

    print("Generation complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Max-3SAT problem instances (Balanced or No Triangle).")
    parser.add_argument("n_vars", type=int, help="Number of variables per instance")
    parser.add_argument("num_instances", type=int, help="Number of instances to generate")
    parser.add_argument("--type", type=str, choices=["balanced", "triangle"], default="balanced", help="Type of instances to generate")
    parser.add_argument("--seed", type=int, default=2000, help="Random seed")
    
    args = parser.parse_args()
    
    generate_dataset(args.n_vars, args.num_instances, args.type, args.seed)
