import random
import os
from pathlib import Path

def sort_clause(clause):
    """
    Sort a clause canonically for uniqueness checking.
    Primary sort key: variable index.
    Secondary sort key: positive before negative.
    """
    def sort_key(lit):
        var = abs(lit)
        return (var, 0 if lit > 0 else 1)
    return tuple(sorted(clause, key=sort_key))

def generate_random_max3sat_instance(num_variables, clause_to_variable_ratio, seed=None, instance_idx=0, num_instances=1):
    """
    Generates a mathematically pure uniform random Max-3SAT instance.
    Enforces constraints:
    1. No repeated variables inside a single clause (e.g. x1 v x1 or x1 v ~x1).
    2. No repeated clauses in the formula.
    """
    ratio = clause_to_variable_ratio if clause_to_variable_ratio is not None else 4.3
    num_clauses = int(num_variables * ratio)
    
    # Create output directory
    output_dir = Path("dataset/satqubolib/randomsat")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set random seed if provided
    instance_seed = seed + instance_idx if seed is not None else None
    if instance_seed is not None:
        random.seed(instance_seed)
        
    seen_clauses = set()
    formula_clauses = []
    
    while len(seen_clauses) < num_clauses:
        # Sample exactly 3 unique variable indices.
        # random.sample guarantees no duplicates inside the list.
        vars = random.sample(range(1, num_variables + 1), 3)
        
        # Apply a uniform 50% chance for each to be negative
        clause = []
        for v in vars:
            sign = 1 if random.random() < 0.5 else -1
            clause.append(sign * v)
            
        canonical_clause = sort_clause(clause)
        
        # If mathematically unique, add to the formula
        if canonical_clause not in seen_clauses:
            seen_clauses.add(canonical_clause)
            formula_clauses.append(clause)  # We can output the unsorted clause to DIMACS
            
    # Construct filename
    suffix = f"_seed{instance_seed}" if seed is not None else f"_{instance_idx+1}"
    output_filename = f"sat_{num_variables}_vars_{num_clauses}_clauses{suffix}.cnf"
    
    output_path = output_dir / output_filename
    
    # Write directly to DIMACS CNF format
    with open(output_path, 'w') as f:
        f.write(f"p cnf {num_variables} {num_clauses}\n")
        for c in formula_clauses:
            f.write(f"{c[0]} {c[1]} {c[2]} 0\n")
            
    return str(output_path), instance_seed
