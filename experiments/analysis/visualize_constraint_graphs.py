
import os
import sys
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path

# Ensure we can import satqubolib
# Assuming running from project root or experiments/, we need to verify import
# Try adding src/dataset to path if satqubolib is local, but it seems to be installed in venv.

try:
    from satqubolib.generators import BalancedSAT, NoTriangleSAT
    from satqubolib.formula import CNF
except ImportError:
    # If strictly local in src/dataset (unlikely based on imports), add path
    current_dir = Path(__file__).parent.absolute()
    project_root = current_dir.parent.parent
    sys.path.append(str(project_root / "src" / "dataset")) # Just in case
    try:
        from satqubolib.generators import BalancedSAT, NoTriangleSAT
        from satqubolib.formula import CNF
    except ImportError:
        print("Error: Could not import satqubolib. Make sure you are running in the correct venv.")
        sys.exit(1)

def build_interaction_graph(cnf: CNF, n_vars: int):
    """
    Builds a variable interaction graph from a CNF formula.
    Nodes: Variables (1 to N)
    Edges: (u, v) if u and v appear in the same clause.
    """
    G = nx.Graph()
    # Add nodes
    for i in range(1, n_vars + 1):
        G.add_node(i)
    
    # Add edges from clauses
    for clause in cnf.clauses:
        # clause is a list of integers (literals)
        vars_in_clause = [abs(lit) for lit in clause]
        # Connect all pairs
        for i in range(len(vars_in_clause)):
            for j in range(i + 1, len(vars_in_clause)):
                u, v = vars_in_clause[i], vars_in_clause[j]
                if u != v:
                    G.add_edge(u, v)
    return G

def analyze_structure(name, cnf, G):
    """
    Calculates and prints statistics for the CNF and its interaction graph.
    """
    print(f"\n--- Statistics for {name} ---")
    
    # 1. Variable Occurrences & Negations
    var_counts = {}
    neg_counts = {}
    
    for clause in cnf.clauses:
        for lit in clause:
            var = abs(lit)
            var_counts[var] = var_counts.get(var, 0) + 1
            if lit < 0:
                neg_counts[var] = neg_counts.get(var, 0) + 1
                
    avg_occurrence = sum(var_counts.values()) / len(var_counts) if var_counts else 0
    print(f"Variable Occurrences: Avg={avg_occurrence:.2f}, Min={min(var_counts.values())}, Max={max(var_counts.values())}")
    
    total_literals = sum(var_counts.values())
    total_negations = sum(neg_counts.values())
    neg_ratio = total_negations / total_literals if total_literals > 0 else 0
    print(f"Negation Ratio: {neg_ratio:.2%}")

    # 2. Clause Statistics
    clause_lengths = [len(c) for c in cnf.clauses]
    avg_len = sum(clause_lengths) / len(clause_lengths) if clause_lengths else 0
    print(f"Clause Lengths: Avg={avg_len:.2f}, Min={min(clause_lengths)}, Max={max(clause_lengths)}")
    
    # Check for unique variable sets (ignoring negation)
    start_sets = [frozenset([abs(l) for l in c]) for c in cnf.clauses]
    unique_sets = set(start_sets)
    print(f"Total Clauses: {len(cnf.clauses)}")
    print(f"Unique Variable Sets: {len(unique_sets)}")
    print(f"Total Clauses: {len(cnf.clauses)}")
    print(f"Unique Variable Sets: {len(unique_sets)}")
    
    # Analyze combinations per variable set
    var_set_counts = {} # frozenset(vars) -> count of clauses (sign combos)
    for c in cnf.clauses:
        vset = frozenset([abs(l) for l in c])
        var_set_counts[vset] = var_set_counts.get(vset, 0) + 1
        
    # Distribution of counts
    count_distribution = {} # count -> how many sets have this count
    for count in var_set_counts.values():
        count_distribution[count] = count_distribution.get(count, 0) + 1
        
    print("\nClause Combinations per Variable Set:")
    for k in sorted(count_distribution.keys()):
        print(f"  {count_distribution[k]} sets have {k} different sign combinations")
        
    if len(unique_sets) < len(cnf.clauses):
        print(f"  (Redundancy: {len(cnf.clauses) - len(unique_sets)} clauses share variable sets with others)")

    # 2. Graph Statistics
    num_triangles = sum(nx.triangles(G).values()) // 3
    density = nx.density(G)
    print(f"Graph Density: {density:.2f}")
    print(f"Number of Triangles: {num_triangles}")
    
    return num_triangles

def main():
    N = 12
    # Ratio 4.3 is standard for Balanced 3-SAT phase transition
    M = int(N * 4.3)
    
    # Output dirs
    temp_dir = Path("dataset/visualization_temp")
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    graph_dir = Path("results/graphs")
    graph_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating instances with N={N}, M={M}...")

    # 1. Generate Balanced SAT
    print("Generating Balanced SAT...")
    gen_balanced = BalancedSAT(N, M, vars_per_clause=3)
    cnf_balanced = gen_balanced.generate()
    cnf_balanced.to_file(str(temp_dir / "balanced_vis.cnf"))
    
    # 2. Generate Triangle-Free SAT
    print("Generating 'Triangle-Free' SAT...")
    gen_triangle = NoTriangleSAT(N, M, vars_per_clause=3) 
    cnf_triangle = gen_triangle.generate()
    cnf_triangle.to_file(str(temp_dir / "triangle_vis.cnf"))

    # 3. Build Graphs
    G_balanced = build_interaction_graph(cnf_balanced, N)
    G_triangle = build_interaction_graph(cnf_triangle, N)
    
    # 4. Analyze
    tri_balanced = analyze_structure("Balanced SAT", cnf_balanced, G_balanced)
    tri_triangle = analyze_structure("Triangle-Free SAT", cnf_triangle, G_triangle)
    
    # 5. Visualize
    plt.figure(figsize=(15, 7))
    
    # Plot Balanced
    plt.subplot(1, 2, 1)
    pos_b = nx.spring_layout(G_balanced, seed=42)
    nx.draw(G_balanced, pos_b, with_labels=True, node_color='lightblue', 
            node_size=500, font_weight='bold', edge_color='gray')
    plt.title(f"Balanced SAT (N={N}, M={M})\nTriangles: {tri_balanced}")
    
    # Plot Triangle-Free
    plt.subplot(1, 2, 2)
    pos_t = nx.spring_layout(G_triangle, seed=42)
    nx.draw(G_triangle, pos_t, with_labels=True, node_color='lightgreen', 
            node_size=500, font_weight='bold', edge_color='gray')
    plt.title(f"Triangle-Free SAT (N={N}, M={M})\nTriangles: {tri_triangle}")
    
    output_path = graph_dir / "constraint_comparison.png"
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"\nVisualization saved to {output_path}")

if __name__ == "__main__":
    main()
