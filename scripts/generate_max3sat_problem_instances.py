
import os
import sys
import subprocess
import random
import argparse

def generate_balanced_dataset(n_vars, num_instances, seed):
    # Locate paths relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(script_dir, "..")
    dataset_script = os.path.join(project_root, "src", "dataset", "satqubolib_max3sat.py")
    
    # Virtual environment python
    venv_python = os.path.join(project_root, "venv", "bin", "python3")
    
    if not os.path.exists(venv_python):
        print(f"Error: Virtual environment python not found at {venv_python}")
        sys.exit(1)

    if not os.path.exists(dataset_script):
        print(f"Error: Dataset generation script not found at {dataset_script}")
        sys.exit(1)

    # Base ratio for balanced instances
    base_ratio = 3.6
    type_name = "balanced"

    # Seed RNG
    random.seed(seed)

    print(f"Generating {num_instances} balanced instances for N={n_vars}...")

    for i in range(num_instances):
        # Randomize ratio +/- 20%
        # random() gives [0, 1). We want [-0.2, 0.2].
        variation = (random.random() * 0.4) - 0.2
        instance_ratio = base_ratio * (1.0 + variation)
        
        # Construct command
        cmd = [
            venv_python,
            dataset_script,
            str(n_vars),
            str(instance_ratio),
            "1", # Generate 1 instance per call
            "--seed", str(seed),
            "--type", type_name
        ]
        
        # Execute
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error generating instance {i}: {e}")

    print("Generation complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Max-3SAT problem instances (Balanced).")
    parser.add_argument("n_vars", type=int, help="Number of variables per instance")
    parser.add_argument("num_instances", type=int, help="Number of instances to generate")
    parser.add_argument("--seed", type=int, default=2000, help="Random seed")
    
    args = parser.parse_args()
    
    generate_balanced_dataset(args.n_vars, args.num_instances, args.seed)
