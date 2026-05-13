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
    with RC2(wcnf) as rc2:
        # Standard enumerate() in RC2 might return all models (even suboptimal).
        # To strictly enumerate OPTIMAL models, we use a blocking clause strategy.
        
        models = []
        model = rc2.compute()
        
        # If no solution found at all (shouldn't pass without hard clauses)
        if not model:
            raise RuntimeError("RC2 could not find a solution.")

        min_cost = rc2.cost
        
        while model is not None:
             # Stop if we drifted into higher cost solutions
             if rc2.cost > min_cost:
                 break
                 
             models.append(model)
             
             # Add a hard clause blocking this specific assignment
             # To block assignment M, we add clause: OR(literals opposite to M)
             rc2.add_clause([-l for l in model])
             
             model = rc2.compute()
             
        return models, min_cost

