#!/usr/bin/env python
"""
Master script to run the neural network training and generate all plots.
Execute this to regenerate all visualizations.
"""

import subprocess
import sys

print("\n" + "="*70)
print("PENDULUM NEURAL DYNAMICS: FULL PIPELINE")
print("="*70 + "\n")

# Run the main training script
print("Step 1: Running Hamiltonian NN training and plot generation...\n")
try:
    subprocess.run([sys.executable, "generate_plots.py"], check=True)
    print("\n✓ Successfully generated all plots!\n")
except Exception as e:
    print(f"✗ Error running generate_plots.py: {e}\n")
    sys.exit(1)

print("="*70)
print("All visualizations have been generated!")
print("Check the repository for:")
print("  - training_loss_comparison.png")
print("  - trajectory_comparison_ham_vs_std.png")
print("  - energy_conservation_comparison.png")
print("  - energy_error_comparison.png")
print("  - relative_energy_error_comparison.png")
print("  - phase_space_comparison.png")
print("="*70 + "\n")
