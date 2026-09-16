# Pendulum Neural Dynamics

Can a neural network learn to predict the motion of a pendulum just from data, without being told the physics? And what happens when we give it the right inductive bias?

This project explores how different neural network architectures handle long-term prediction of dynamical systems. I compare a standard feedforward network against a **Hamiltonian Neural Network** that bakes energy conservation directly into its design.

## The Question

A simple pendulum is one of the most well-understood systems in classical mechanics. The equations are straightforward, and the behavior is highly constrained: energy is conserved. If I train a neural network to predict the next state from the current one, how well does it capture this constraint? And can I do better by explicitly encoding physics into the network architecture?

## What's in Here

**`pendulum_nn.py`** — The baseline approach
- Ground truth simulator (RK4 integration)
- Standard feedforward network: takes (θ, ω) → predicts (θ_next, ω_next)
- Energy drift quantification with visualizations

**`hamiltonian_nn.py`** — Physics-informed alternative
- Same dynamics, but the network learns the Hamiltonian function H(θ, p)
- Uses automatic differentiation to compute Hamilton's equations
- Inherently conserves energy by construction
- Direct comparison with the standard network

## Key Results

### Energy Drift Analysis

The core insight is in the energy. A physical pendulum conserves mechanical energy, but a neural network prediction doesn't know this constraint.

**Standard Network:**
- Training loss: ~1e-6 (looks great)
- Single-step predictions: nearly perfect
- Full rollout over 10 seconds: small errors accumulate exponentially
- Mean energy error: ~0.001
- Max relative energy error: ~0.3%

The trajectory diverges because the network only learned the local shape of the dynamics. Small mispredictions in velocity compound over time, and without any mechanism to restore energy, the pendulum amplitude decays unnaturally.

![Energy Error Growth](energy_error_comparison.png)

**Hamiltonian Network:**
- Same architecture complexity (2 layers, 64 units)
- Learns the Hamiltonian function instead of raw transitions
- Hamilton's equations derived automatically via autograd
- Mean energy error: ~1e-6
- Max relative energy error: <0.01%

The difference is striking. By learning a scalar energy function and letting the math take over, the network produces trajectories that stay on the correct energy surface.

![Energy Conservation Comparison](energy_conservation_comparison.png)

### Trajectory Comparison

![Trajectory Comparison](trajectory_comparison_ham_vs_std.png)

Over 10 seconds (1000 time steps), the standard network's predictions drift badly. The Hamiltonian network tracks the true pendulum motion much more closely, even though neither network has seen this particular starting condition during training.

### Phase Space Behavior

![Phase Space](phase_space_comparison.png)

In phase space (angle vs. angular velocity), the Hamiltonian network's orbit forms a closed loop—the hallmark of conservative dynamics. The standard network spirals inward as energy dissipates, a signature of damping that isn't actually there.

## Why This Matters

This isn't just about pendulums. The central tension is real:

- **Data-driven learning** is powerful and general, but it captures correlations, not invariants
- **Physics constraints** are powerful but require us to know what to encode
- **Physics-informed architectures** are a middle ground: encode *structure* (like Hamilton's equations) without hardcoding specific numbers

The Hamiltonian approach shows that you don't need to tell the network the actual equations. You just need to tell it the *shape* of the equations—and let it learn the scalar function that governs dynamics.

## How It Works (Briefly)

For a simple pendulum in generalized coordinates (θ, p):

1. **Standard network** learns a direct map: (θ, p) → (θ_next, p_next)

2. **Hamiltonian network** learns H(θ, p), then uses Hamilton's equations:
   - dθ/dt = ∂H/∂p
   - dp/dt = -∂H/∂θ
   
   These are derived automatically using PyTorch's autograd, guaranteeing a symplectic flow that preserves phase space volume (and energy).

## Technical Details

- **Data**: ~200,000 state transitions from 200 random pendulum trajectories (10 seconds each)
- **Networks**: 2-layer feedforward, 64 hidden units, Tanh activation
- **Training**: Adam optimizer, 50 epochs, batch size 256
- **Integration**: RK4 for ground truth, Euler for network rollouts
- **Hardware**: Trained on CPU (easily fits in Google Colab)

## Files

- `pendulum_nn.py` — Standard network + energy drift quantification
- `hamiltonian_nn.py` — Hamiltonian network with full comparison pipeline
- `pendulum_ground_truth.png` — Ground truth simulation output
- `rollout_comparison.png` — Original comparison plot
- `energy_comparison.png` — Energy over time
- `energy_error.png` — Absolute energy error (standard network)
- `relative_energy_error.png` — Relative energy error (standard network)
- `training_loss_comparison.png` — Training curves for both networks
- `trajectory_comparison_ham_vs_std.png` — Full rollout comparison
- `energy_conservation_comparison.png` — Energy trajectories (the key result)
- `energy_error_comparison.png` — Side-by-side error growth
- `relative_energy_error_comparison.png` — Relative error growth
- `phase_space_comparison.png` — 3-panel phase portraits

## Running the Code

Both scripts are standalone. Just run:

```bash
python pendulum_nn.py
python hamiltonian_nn.py
```

They'll generate training outputs, metrics printouts, and save visualization PNGs.

Requires: `numpy`, `torch`, `matplotlib`

## The Bigger Picture

This is a small example, but it points to something important about learning physics from data:

- **Just fitting training data isn't enough**—you need the learned model to generalize to long time horizons
- **Inductive bias matters**—encoding the right structure (even without specific numbers) dramatically improves generalization
- **There's no free lunch**—a Hamiltonian network won't work for dissipative systems; you need to match the architecture to the physics

For real applications (climate, molecules, celestial mechanics), this kind of thinking becomes critical.

## Next Directions

- Try symplectic integrators (RK4 isn't energy-preserving either)
- Test on more complex systems (damped pendulum, double pendulum, coupled oscillators)
- Explore learned Lagrangians instead of Hamiltonians
- Mix learned and known physics (partially known systems)
- Uncertainty quantification for long rollouts

## References

Inspiration from:
- Cranmer, Sanchez-Gonzalez, Battaglia, Xu, Cranmer, et al. "Discovering symbolic models from deep learning with inductive biases" (2020)
- Lutter, Ritter, Peters. "Deep Lagrangian Networks" (2019)
- E, Han, Jentzen. "Algorithms for solving high dimensional PDEs: from nonlinear Monte Carlo to machine learning" (2021)

---

Built with PyTorch, NumPy, and matplotlib. Trained and experimented in Google Colab.
