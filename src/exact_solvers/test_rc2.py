import os
import subprocess
import glob
import sys
from rc2 import solve_rc2

def test_rc2_dynamic():
    """
    Tests the solve_rc2 function by dynamically generating a Max3SAT instance.
    """
    # 1. Define instance parameters
    n_vars = 10
    ratio = 4.24
    n_instances = 1
    seed = 123
    instance_type = "balanced"
    
    # Path to the generator script
    # Assuming this script is running from the project root or src/exact_solvers
    # We'll adjust paths relative to project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    generator_script = os.path.join(project_root, "src/dataset/satqubolib_max3sat.py")
    
    # 2. Generate the instance
    cmd = [
        sys.executable, generator_script,
        str(n_vars), str(ratio), str(n_instances),
        "--seed", str(seed),
        "--type", instance_type
    ]
    
    print(f"Running generator: {' '.join(cmd)}")
    subprocess.check_call(cmd, cwd=project_root)
    
    # 3. Find the generated file
    # It should be in dataset/satqubolib/balancedsat/
    dataset_dir = os.path.join(project_root, "dataset/satqubolib/balancedsat")
    # Expected filename pattern based on generator logic (or just find the latest/only one)
    # The generator produces: sat_{n_vars}_vars_{ratio}_ratio_seed{seed}.cnf
    # Note: ratios might be formatted. Let's look for *seed123.cnf
    search_pattern = os.path.join(dataset_dir, f"*seed{seed}.cnf")
    files = glob.glob(search_pattern)
    
    if not files:
        raise FileNotFoundError(f"No generated instance found for seed {seed}")
    
    instance_path = files[0]
    print(f"Testing with instance: {instance_path}")
    
    try:
        # 4. Solves the instance
        models, cost = solve_rc2(instance_path)
        
        print("-" * 20)
        print(f"Number of optimal models found: {len(models)}")
        # print first 5 models to avoid spam
        for i, m in enumerate(models[:5]):
            print(f"Model {i+1}: {m}")
        if len(models) > 5:
            print("...")
        print(f"Cost: {cost}")
        print("-" * 20)
        
        # 5. Assertions
        # Cost should be non-negative.
        assert cost >= 0, "Cost must be non-negative"
        
        # Models should be a non-empty list
        assert isinstance(models, list), "models should be a list"
        assert len(models) > 0, "Should find at least one model"
        
        # Check first model structure
        first_model = models[0]
        assert len(first_model) == n_vars, f"Model should have {n_vars} variables, got {len(first_model)}"
        
        # We can loosely check if the model is valid by checking if it contains integers from 1 to 10 (abs values)
        abs_model = sorted([abs(x) for x in first_model])
        assert abs_model == list(range(1, n_vars + 1)), "Model variables should match instance variables"

        print("Test passed successfully!")

    finally:
        # 6. Cleanup
        if os.path.exists(instance_path):
            print(f"Removing file: {instance_path}")
            os.remove(instance_path)

if __name__ == "__main__":
    test_rc2_dynamic()
