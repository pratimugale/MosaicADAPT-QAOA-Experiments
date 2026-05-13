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

def generate_max3sat_instance(sat_type, num_variables, clause_to_variable_ratio, seed=None, instance_idx=0, num_instances=1):
    """
    Generates a single Max-3SAT instance (balanced or triangle-free) and saves it to DIMACS format.
    """
    # Determine parameters based on type
    if sat_type == 'balanced':
        ratio = clause_to_variable_ratio if clause_to_variable_ratio is not None else 3.6
        num_clauses = int(num_variables * ratio)
        gen_class = BalancedSAT
        dir_name = "balancedsat"
    elif sat_type == 'triangle':
        ratio = clause_to_variable_ratio if clause_to_variable_ratio is not None else 4.3
        num_clauses = int(num_variables * ratio)
        gen_class = NoTriangleSAT
        dir_name = "notrianglesat"
    else:
        raise ValueError(f"Unknown SAT type: {sat_type}")

    # Create output directory
    output_dir = Path("dataset/satqubolib") / dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set random seed if provided
    instance_seed = seed + instance_idx if seed is not None else None
    if instance_seed is not None:
        random.seed(instance_seed)
        np.random.seed(instance_seed)
    
    generator = gen_class(num_variables, num_clauses, vars_per_clause=3)
    
    # Check for duplicate clauses within the generated formula
    seen_clauses = set()
    unique_clauses = []
    
    while len(unique_clauses) < num_clauses:
        # Generate a batch of clauses
        cnf_formula = generator.generate()
        
        for clause in cnf_formula.clauses:
            if len(unique_clauses) >= num_clauses:
                break
                
            # Sort the literals within the clause to ensure canonical representation
            def sort_key(lit):
                var = abs(lit)
                return (var, 0 if lit > 0 else 1)
                
            canonical_clause = tuple(sorted(clause, key=sort_key))
            
            if canonical_clause not in seen_clauses:
                seen_clauses.add(canonical_clause)
                unique_clauses.append(clause)
                
    # Rebuild the CNF formula with strictly unique clauses
    final_formula = CNF(unique_clauses)
            
    # Construct filename
    prefix = f"{sat_type}_"
    suffix = f"_seed{instance_seed}" if seed is not None else f"_{instance_idx+1}"
    output_filename = f"{prefix}sat_{num_variables}_vars_{num_clauses}_clauses{suffix}.cnf"
    
    output_path = output_dir / output_filename
    final_formula.to_file(str(output_path))
    return output_path, instance_seed

def main():
    parser = argparse.ArgumentParser(description='Generate max3sat instances using satqubolib')
    parser.add_argument('--type', type=str, choices=['balanced', 'triangle'], default='balanced',
                       help='Type of SAT instance to generate: balanced or triangle (default: balanced)')
    parser.add_argument('num_variables', type=int, nargs='?', 
                       help='Number of variables (for balanced) or nodes (for triangle)')
    parser.add_argument('clause_to_variable_ratio', type=float, nargs='?', default=None,
                       help='Clause to variable ratio (default: 3.6 for balanced, 4.3 for triangle)')
    parser.add_argument('num_instances', type=int, help='Number of instances to generate')
    parser.add_argument('--seed', type=int, default=None, 
                       help='Random seed for reproducibility (optional). If provided, each instance will use seed + instance_index')
    
    args = parser.parse_args()
    
    if args.num_variables is None:
        parser.error("num_variables (or num_nodes for triangle) is required")
        
    for i in range(args.num_instances):
        path, used_seed = generate_max3sat_instance(
            args.type, 
            args.num_variables, 
            args.clause_to_variable_ratio, 
            seed=args.seed, 
            instance_idx=i, 
            num_instances=args.num_instances
        )
        seed_msg = f" (seed: {used_seed})" if used_seed is not None else ""
        print(f"Instance {i+1}/{args.num_instances} saved to: {path}{seed_msg}")

    print(f"Successfully generated {args.num_instances} instance(s)")

if __name__ == "__main__":
    main()