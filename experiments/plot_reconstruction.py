
import json
import matplotlib.pyplot as plt
import os
import sys

# Load Data
results_file = os.path.join(os.path.dirname(__file__), "..", "results", "reconstruction_data.json")

print(f"Loading data from {results_file}...")
with open(results_file, 'r') as f:
    data = json.load(f)

# Extract
adaptations = data["adaptations"]
max_sat = data["max_sat"]
energy = data["energy"]

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Best SAT
axes[0].plot(adaptations, max_sat, marker='o', linestyle='-', color='b')
axes[0].set_title("Best Satisfied Clauses vs. Adaptations")
axes[0].set_xlabel("Adaptation Step")
axes[0].set_ylabel("Best Satisfaction (Clauses)")
axes[0].grid(True)
# Add values on points
for x, y in zip(adaptations, max_sat):
    axes[0].annotate(f"{y}", (x, y), textcoords="offset points", xytext=(0,10), ha='center')

# Plot 2: Energy
axes[1].plot(adaptations, energy, marker='o', linestyle='-', color='r')
axes[1].set_title("Energy vs. Adaptations")
axes[1].set_xlabel("Adaptation Step")
axes[1].set_ylabel("Energy (Expectation)")
axes[1].grid(True)

plt.tight_layout()

# Save
output_path = os.path.join(os.path.dirname(__file__), "analysis", "plots", "reconstruction_10q.png")
# Ensure directory exists
os.makedirs(os.path.dirname(output_path), exist_ok=True)

plt.savefig(output_path)
print(f"Plot saved to {output_path}")
