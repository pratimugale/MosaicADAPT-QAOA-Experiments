# This script is used to generate max3sat instances using the satqubolib library
# The script is called using make generate_satqubolib

import argparse
import os
import random
import sys
from pathlib import Path
import numpy as np
from satqubolib.generators import BalancedSAT, NoTriangleSAT
from satqubolib.formula import CNF

def main():
    parser = argparse.ArgumentParser(description='Generate max3sat instances using satqubolib')
    parser.add_argument('--type', type=str, choices=['balanced', 'triangle'], default='balanced',
                       help='Type of SAT instance to generate: balanced or triangle (default: balanced)')
    parser.add_argument('num_variables', type=int, nargs='?', 
                       help='Number of variables (for balanced) or nodes (for triangle)')
    parser.add_argument('clause_to_variable_ratio', type=float, nargs='?',
                       help='Clause to variable ratio (required for balanced, ignored for triangle)')
    parser.add_argument('num_instances', type=int, help='Number of instances to generate')
    parser.add_argument('--seed', type=int, default=None, 
                       help='Random seed for reproducibility (optional). If provided, each instance will use seed + instance_index')
    
    args = parser.parse_args()
    
    sat_type = args.type
    num_instances = args.num_instances
    seed = args.seed
    
    if args.num_variables is None:
        parser.error("num_variables (or num_nodes for triangle) is required")
    
    if sat_type == 'balanced':
        if args.clause_to_variable_ratio is None:
            parser.error("clause_to_variable_ratio is required for balanced SAT")
        num_variables = args.num_variables
        clause_to_variable_ratio = args.clause_to_variable_ratio
        num_clauses = int(num_variables * clause_to_variable_ratio)
        k_value = 3  # 3-SAT
        
        # Create output directory
        output_dir = Path("dataset/satqubolib/balancedsat")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        seed_info = f" (seed: {seed})" if seed is not None else " (random seed)"
        print(f"Generating {num_instances} instance(s) of {k_value}-SAT (balanced) with {num_variables} variables and {num_clauses} clauses (ratio: {clause_to_variable_ratio}){seed_info}")
        
        for i in range(num_instances):
            # 1. Set random seed if provided (for both random and numpy)
            instance_seed = seed + i if seed is not None else None
            if instance_seed is not None:
                random.seed(instance_seed)
                np.random.seed(instance_seed)
            
            # 2. Instantiate the generator
            # Arguments: (num_vars, num_clauses, vars_per_clause=3)
            generator = BalancedSAT(num_variables, num_clauses, vars_per_clause=k_value) 
            
            # 3. Generate the CNF formula object
            cnf_formula: CNF = generator.generate()
            
            # 4. Save the output to a .cnf file (removed "balanced" prefix)
            if num_instances == 1:
                if seed is not None:
                    output_filename = f"sat_{num_variables}_vars_{clause_to_variable_ratio}_ratio_seed{seed}.cnf"
                else:
                    output_filename = f"sat_{num_variables}_vars_{clause_to_variable_ratio}_ratio.cnf"
            else:
                if seed is not None:
                    output_filename = f"sat_{num_variables}_vars_{clause_to_variable_ratio}_ratio_seed{instance_seed}.cnf"
                else:
                    output_filename = f"sat_{num_variables}_vars_{clause_to_variable_ratio}_ratio_{i+1}.cnf"
            
            output_path = output_dir / output_filename
            
            # Save CNF to file in DIMACS format
            cnf_formula.to_file(str(output_path))
            
            seed_msg = f" (seed: {instance_seed})" if instance_seed is not None else ""
            print(f"Instance {i+1}/{num_instances} saved to: {output_path}{seed_msg}")
    
    elif sat_type == 'triangle':
        num_nodes = args.num_variables
        # For NoTriangleSAT, we need num_vars and num_clauses
        # Using num_nodes as num_vars, and calculating num_clauses based on graph structure
        # For a graph with n nodes, we typically need clauses proportional to possible edges
        # Using a reasonable default: approximately 4.3 * num_nodes (similar to balanced SAT ratio)
        num_vars = num_nodes
        num_clauses = int(num_nodes * 4.3)  # Default clause-to-variable ratio
        
        # Create output directory
        output_dir = Path(f"dataset/satqubolib/notrianglesat/{num_nodes}nodes")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        seed_info = f" (seed: {seed})" if seed is not None else " (random seed)"
        print(f"Generating {num_instances} instance(s) of NoTriangle SAT with {num_nodes} nodes and {num_clauses} clauses{seed_info}")
        
        for i in range(num_instances):
            # 1. Set random seed if provided (for both random and numpy)
            instance_seed = seed + i if seed is not None else None
            if instance_seed is not None:
                random.seed(instance_seed)
                np.random.seed(instance_seed)
            
            # 2. Instantiate the generator
            # Arguments: (num_vars, num_clauses, vars_per_clause=3)
            generator = NoTriangleSAT(num_vars, num_clauses, vars_per_clause=3)
            
            # 3. Generate the CNF formula object
            cnf_formula: CNF = generator.generate()
            
            # 4. Save the output to a .cnf file
            if num_instances == 1:
                if seed is not None:
                    output_filename = f"sat_{num_nodes}_nodes_seed{seed}.cnf"
                else:
                    output_filename = f"sat_{num_nodes}_nodes.cnf"
            else:
                if seed is not None:
                    output_filename = f"sat_{num_nodes}_nodes_seed{instance_seed}.cnf"
                else:
                    output_filename = f"sat_{num_nodes}_nodes_{i+1}.cnf"
            
            output_path = output_dir / output_filename
            
            # Save CNF to file in DIMACS format
            cnf_formula.to_file(str(output_path))
            
            seed_msg = f" (seed: {instance_seed})" if instance_seed is not None else ""
            print(f"Instance {i+1}/{num_instances} saved to: {output_path}{seed_msg}")
    
    print(f"Successfully generated {num_instances} instance(s)")

if __name__ == "__main__":
    main()