from pysat.examples.rc2 import RC2
from pysat.formula import CNF, WCNF

def solve_rc2(instance_path: str) -> tuple[list[list[int]], int]:
    """
    Solves a MaxSAT instance using the RC2 solver.

    Args:
        instance_path (str): Path to the DIMACS CNF file.

    Returns:
        tuple[list[list[int]], int]: A tuple containing a list of optimal models
                                     (each model is a list of integers) and the cost.
    """
    # 1. Load the CNF formula
    cnf = CNF(from_file=instance_path)

    # 2. Create a WCNF formula for MaxSAT
    # Soft clauses have weight > 0 (here 1 for unweighted MaxSAT)
    # Hard clauses would have top weight, but for standard Max3SAT all are soft
    wcnf = WCNF()
    
    # Add all clauses from CNF as soft clauses with weight 1
    for clause in cnf.clauses:
        wcnf.append(clause, weight=1)

    # 3. Use the RC2 solver

    rc2 = RC2(wcnf)
    
    # Find all optimal solutions
    models = []
    optimal_cost = None
    
    # Loop to find multiple optimal solutions
    while True:
        model = rc2.compute()
        
        if model is None:
            break
            
        cost = rc2.cost
        
        if optimal_cost is None:
            optimal_cost = cost
        elif cost > optimal_cost:
            break
            
        # IMPORTANT: Store a COPY of the model, as pysat reuses the reference
        models.append(list(model))
        
        # Block this solution to find others
        rc2.add_clause([-l for l in model])
        
    return models, optimal_cost
        

