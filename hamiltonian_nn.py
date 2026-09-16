# -*- coding: utf-8 -*-
"""
Hamiltonian Neural Network for Pendulum Dynamics

A physics-informed neural network that learns to predict pendulum dynamics
while explicitly enforcing energy conservation through the Hamiltonian formalism.

Key idea: Instead of learning the raw state transitions directly, learn the
Hamiltonian H(theta, p) where p is the generalized momentum. The equations of
motion are then derived automatically from Hamilton's equations:
    dtheta/dt = dH/dp
    dp/dt = -dH/dtheta

This architectural choice ensures energy conservation by construction.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Physical constants
g = 9.81      # gravity (m/s^2)
L = 1.0       # pendulum length (m)

# ============================================================================
# GROUND TRUTH SIMULATOR (RK4)
# ============================================================================

def pendulum_deriv(state, t):
    """Compute derivatives for pendulum using standard physics."""
    theta, omega = state
    dtheta_dt = omega
    domega_dt = -(g / L) * np.sin(theta)
    return np.array([dtheta_dt, domega_dt])

def simulate(theta0, omega0, t_max=10.0, dt=0.01):
    """Simulate pendulum using RK4 integration."""
    steps = int(t_max / dt)
    states = np.zeros((steps, 2))
    states[0] = [theta0, omega0]

    for i in range(1, steps):
        s = states[i-1]
        k1 = pendulum_deriv(s, 0)
        k2 = pendulum_deriv(s + dt/2 * k1, 0)
        k3 = pendulum_deriv(s + dt/2 * k2, 0)
        k4 = pendulum_deriv(s + dt * k3, 0)
        states[i] = s + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

    return states

# ============================================================================
# HAMILTONIAN NEURAL NETWORK
# ============================================================================

class HamiltonianNet(nn.Module):
    """
    A neural network that learns the Hamiltonian H(theta, p).
    
    The Hamiltonian for a simple pendulum is:
        H(theta, p) = p^2 / (2*m*L^2) + m*g*L*(1 - cos(theta))
    
    where p is the generalized momentum = m*L^2*omega.
    
    Hamilton's equations give us:
        dtheta/dt = dH/dp
        dp/dt = -dH/dtheta
    
    By learning H, we automatically get energy-conserving dynamics.
    """
    
    def __init__(self, hidden_size=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)  # Output is a scalar (Hamiltonian)
        )
    
    def forward(self, q):
        """
        Compute Hamiltonian H(q) where q = [theta, p].
        
        Args:
            q: (batch_size, 2) tensor of [theta, p]
        
        Returns:
            H: (batch_size, 1) tensor of Hamiltonian values
        """
        return self.net(q)
    
    def hamiltonian_dynamics(self, q, dt=0.01):
        """
        Compute next state using Hamilton's equations with autograd.
        
        Hamilton's equations:
            dq/dt = dH/dp = [dH/dp, -dH/dtheta]
        
        We use automatic differentiation to compute gradients of H.
        Then integrate using Euler step: q_next = q + dt * dq/dt
        
        Args:
            q: (batch_size, 2) tensor of [theta, p]
            dt: timestep
        
        Returns:
            q_next: (batch_size, 2) tensor of next state
        """
        q = q.clone().detach().requires_grad_(True)
        
        # Compute Hamiltonian
        H = self.forward(q)
        H_sum = H.sum()
        
        # Compute gradients: dH/dq
        H_sum.backward()
        dH_dq = q.grad  # (batch_size, 2)
        
        # Hamilton's equations: dq/dt = J @ dH/dq
        # where J is the symplectic matrix [[0, 1], [-1, 0]]
        # So: dtheta/dt = dH/dp, dp/dt = -dH/dtheta
        dtheta_dt = dH_dq[:, 1:2]  # dH/dp
        dp_dt = -dH_dq[:, 0:1]      # -dH/dtheta
        
        dq_dt = torch.cat([dtheta_dt, dp_dt], dim=1)
        
        # Euler integration
        q_next = q.detach() + dt * dq_dt.detach()
        
        return q_next


class StandardNet(nn.Module):
    """
    Standard feedforward network (for comparison).
    Predicts next state directly without energy conservation constraint.
    """
    
    def __init__(self, hidden_size=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 2)
        )
    
    def forward(self, x):
        return self.net(x)


# ============================================================================
# DATA GENERATION (generalized coordinates)
# ============================================================================

def generate_hamiltonian_dataset(n_trajectories=200, t_max=10.0, dt=0.01):
    """
    Generate training data in generalized coordinates (theta, p).
    
    p = m * L^2 * omega, where we set m*L^2 = 1 for simplicity.
    So p ≈ omega.
    """
    X = []  # current state (theta, p)
    Y = []  # next state (theta, p)

    for _ in range(n_trajectories):
        theta0 = np.random.uniform(-np.pi/2, np.pi/2)
        omega0 = np.random.uniform(-1, 1)
        p0 = omega0  # Normalized: m*L^2 = 1
        
        states = simulate(theta0, omega0, t_max, dt)
        # Convert to generalized coordinates
        states_generalized = np.copy(states)
        states_generalized[:, 1] = states[:, 1]  # p ≈ omega
        
        X.append(states_generalized[:-1])
        Y.append(states_generalized[1:])

    X = np.vstack(X)
    Y = np.vstack(Y)
    return X, Y


def compute_energy(trajectory, g=9.81, L=1.0):
    """
    Compute total mechanical energy for a pendulum.
    
    E = (1/2) * omega^2 + (g/L) * (1 - cos(theta))
    """
    theta = trajectory[:, 0]
    omega = trajectory[:, 1]
    
    kinetic = 0.5 * omega**2
    potential = (g / L) * (1 - np.cos(theta))
    
    return kinetic + potential


# ============================================================================
# TRAINING
# ============================================================================

print("\n" + "="*70)
print("HAMILTONIAN NEURAL NETWORK FOR PENDULUM DYNAMICS")
print("="*70 + "\n")

# Generate dataset
print("Generating training data...")
X, Y = generate_hamiltonian_dataset()
print(f"Dataset shape: {X.shape}, {Y.shape}")

X_tensor = torch.tensor(X, dtype=torch.float32)
Y_tensor = torch.tensor(Y, dtype=torch.float32)

# Initialize both networks
print("\nInitializing networks...")
ham_model = HamiltonianNet(hidden_size=64)
std_model = StandardNet(hidden_size=64)

print("Hamiltonian Network:")
print(ham_model)
print("\nStandard Network:")
print(std_model)

# Training
criterion = nn.MSELoss()
ham_optimizer = optim.Adam(ham_model.parameters(), lr=0.001)
std_optimizer = optim.Adam(std_model.parameters(), lr=0.001)

n_epochs = 50
batch_size = 256
n_samples = X_tensor.shape[0]

ham_losses = []
std_losses = []

print("\n" + "="*70)
print("TRAINING")
print("="*70 + "\n")

for epoch in range(n_epochs):
    perm = torch.randperm(n_samples)
    ham_epoch_loss = 0.0
    std_epoch_loss = 0.0

    for i in range(0, n_samples, batch_size):
        idx = perm[i:i+batch_size]
        x_batch = X_tensor[idx]
        y_batch = Y_tensor[idx]

        # Train Standard Network
        std_optimizer.zero_grad()
        std_pred = std_model(x_batch)
        std_loss = criterion(std_pred, y_batch)
        std_loss.backward()
        std_optimizer.step()
        std_epoch_loss += std_loss.item() * x_batch.shape[0]

        # Train Hamiltonian Network
        ham_optimizer.zero_grad()
        ham_pred = ham_model.hamiltonian_dynamics(x_batch, dt=0.01)
        ham_loss = criterion(ham_pred, y_batch)
        ham_loss.backward()
        ham_optimizer.step()
        ham_epoch_loss += ham_loss.item() * x_batch.shape[0]

    ham_epoch_loss /= n_samples
    std_epoch_loss /= n_samples
    ham_losses.append(ham_epoch_loss)
    std_losses.append(std_epoch_loss)

    if epoch % 5 == 0 or epoch == n_epochs - 1:
        print(f"Epoch {epoch:3d} | Hamiltonian Loss: {ham_epoch_loss:.8f} | Standard Loss: {std_epoch_loss:.8f}")

# ============================================================================
# ROLLOUT AND EVALUATION
# ============================================================================

def rollout_hamiltonian(model, theta0, omega0, steps=999, dt=0.01):
    """Rollout using Hamiltonian dynamics."""
    state = torch.tensor([theta0, omega0], dtype=torch.float32).unsqueeze(0)
    trajectory = [state.numpy().squeeze()]

    with torch.no_grad():
        for _ in range(steps):
            state = model.hamiltonian_dynamics(state, dt=dt)
            trajectory.append(state.numpy().squeeze())

    return np.array(trajectory)


def rollout_standard(model, theta0, omega0, steps=999, dt=0.01):
    """Rollout using standard network."""
    state = torch.tensor([theta0, omega0], dtype=torch.float32)
    trajectory = [state.numpy()]

    with torch.no_grad():
        for _ in range(steps):
            state = model(state.unsqueeze(0)).squeeze(0)
            trajectory.append(state.numpy())

    return np.array(trajectory)


# Test on unseen starting condition
print("\n" + "="*70)
print("EVALUATION ON UNSEEN TEST CONDITION")
print("="*70 + "\n")

theta0_test = np.radians(60)
omega0_test = 0.0

true_traj = simulate(theta0_test, omega0_test)
ham_traj = rollout_hamiltonian(ham_model, theta0_test, omega0_test)
std_traj = rollout_standard(std_model, theta0_test, omega0_test)

# Compute energies
true_energy = compute_energy(true_traj)
ham_energy = compute_energy(ham_traj)
std_energy = compute_energy(std_traj)

# Compute metrics
def compute_metrics(true_E, pred_E, name):
    """Compute and print energy metrics."""
    E0 = true_E[0]
    error = np.abs(pred_E - true_E)
    rel_error = error / np.abs(E0)
    
    print(f"\n{name}:")
    print(f"  Initial energy: {E0:.6f}")
    print(f"  Mean energy error: {np.mean(error):.8f}")
    print(f"  Max energy error: {np.max(error):.8f}")
    print(f"  Mean relative error: {np.mean(rel_error):.6%}")
    print(f"  Max relative error: {np.max(rel_error):.6%}")
    
    return error, rel_error

print("Energy Drift Analysis:")
ham_err, ham_rel = compute_metrics(true_energy, ham_energy, "Hamiltonian Network")
std_err, std_rel = compute_metrics(true_energy, std_energy, "Standard Network")

print("\n" + "="*70 + "\n")

# ============================================================================
# VISUALIZATIONS
# ============================================================================

t = np.linspace(0, 10, len(true_traj))

# Plot 1: Training loss comparison
plt.figure(figsize=(12, 5))
plt.plot(ham_losses, label="Hamiltonian Network", linewidth=2)
plt.plot(std_losses, label="Standard Network", linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Training Loss (MSE)")
plt.title("Training Loss Comparison")
plt.legend()
plt.grid(True)
plt.yscale('log')
plt.savefig("training_loss_comparison.png", dpi=150)
plt.show()

# Plot 2: Trajectory comparison
plt.figure(figsize=(14, 6))
plt.plot(t, np.degrees(true_traj[:, 0]), label="True physics (RK4)", linewidth=2.5, color='black')
plt.plot(t, np.degrees(ham_traj[:, 0]), label="Hamiltonian NN", linestyle='--', linewidth=2, color='green')
plt.plot(t, np.degrees(std_traj[:, 0]), label="Standard NN", linestyle='--', linewidth=2, color='red')
plt.xlabel("Time (s)", fontsize=12)
plt.ylabel("Angle (degrees)", fontsize=12)
plt.title("Pendulum Angle: Hamiltonian vs Standard Network", fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.savefig("trajectory_comparison_ham_vs_std.png", dpi=150)
plt.show()

# Plot 3: Energy conservation comparison
plt.figure(figsize=(14, 6))
plt.plot(t, true_energy, label="True energy (conserved)", linewidth=2.5, color='black')
plt.plot(t, ham_energy, label="Hamiltonian NN energy", linestyle='--', linewidth=2, color='green')
plt.plot(t, std_energy, label="Standard NN energy", linestyle='--', linewidth=2, color='red')
plt.xlabel("Time (s)", fontsize=12)
plt.ylabel("Total Energy (normalized)", fontsize=12)
plt.title("Energy Conservation: Hamiltonian vs Standard Network", fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.savefig("energy_conservation_comparison.png", dpi=150)
plt.show()

# Plot 4: Absolute energy error
plt.figure(figsize=(14, 6))
plt.semilogy(t, np.abs(ham_energy - true_energy), label="Hamiltonian NN", linewidth=2.5, color='green')
plt.semilogy(t, np.abs(std_energy - true_energy), label="Standard NN", linewidth=2.5, color='red')
plt.xlabel("Time (s)", fontsize=12)
plt.ylabel("Absolute Energy Error (log scale)", fontsize=12)
plt.title("Energy Error Accumulation Over Time", fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, which='both', alpha=0.3)
plt.savefig("energy_error_comparison.png", dpi=150)
plt.show()

# Plot 5: Relative energy error
plt.figure(figsize=(14, 6))
plt.semilogy(t, np.abs(ham_energy - true_energy) / np.abs(true_energy[0]), 
             label="Hamiltonian NN", linewidth=2.5, color='green')
plt.semilogy(t, np.abs(std_energy - true_energy) / np.abs(true_energy[0]), 
             label="Standard NN", linewidth=2.5, color='red')
plt.xlabel("Time (s)", fontsize=12)
plt.ylabel("Relative Energy Error (log scale)", fontsize=12)
plt.title("Relative Energy Error Over Time", fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, which='both', alpha=0.3)
plt.savefig("relative_energy_error_comparison.png", dpi=150)
plt.show()

# Plot 6: Phase space comparison
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# True physics
axes[0].plot(true_traj[:, 0], true_traj[:, 1], linewidth=2, color='black')
axes[0].scatter(true_traj[0, 0], true_traj[0, 1], color='green', s=100, marker='o', label='Start', zorder=5)
axes[0].scatter(true_traj[-1, 0], true_traj[-1, 1], color='red', s=100, marker='x', linewidth=3, label='End', zorder=5)
axes[0].set_xlabel("Angle (rad)", fontsize=11)
axes[0].set_ylabel("Angular velocity (rad/s)", fontsize=11)
axes[0].set_title("True Physics (RK4)", fontsize=12)
axes[0].grid(True, alpha=0.3)
axes[0].legend()

# Hamiltonian NN
axes[1].plot(ham_traj[:, 0], ham_traj[:, 1], linewidth=2, color='green')
axes[1].scatter(ham_traj[0, 0], ham_traj[0, 1], color='green', s=100, marker='o', label='Start', zorder=5)
axes[1].scatter(ham_traj[-1, 0], ham_traj[-1, 1], color='red', s=100, marker='x', linewidth=3, label='End', zorder=5)
axes[1].set_xlabel("Angle (rad)", fontsize=11)
axes[1].set_ylabel("Angular velocity (rad/s)", fontsize=11)
axes[1].set_title("Hamiltonian Network", fontsize=12)
axes[1].grid(True, alpha=0.3)
axes[1].legend()

# Standard NN
axes[2].plot(std_traj[:, 0], std_traj[:, 1], linewidth=2, color='red')
axes[2].scatter(std_traj[0, 0], std_traj[0, 1], color='green', s=100, marker='o', label='Start', zorder=5)
axes[2].scatter(std_traj[-1, 0], std_traj[-1, 1], color='red', s=100, marker='x', linewidth=3, label='End', zorder=5)
axes[2].set_xlabel("Angle (rad)", fontsize=11)
axes[2].set_ylabel("Angular velocity (rad/s)", fontsize=11)
axes[2].set_title("Standard Network", fontsize=12)
axes[2].grid(True, alpha=0.3)
axes[2].legend()

plt.tight_layout()
plt.savefig("phase_space_comparison.png", dpi=150)
plt.show()

print("All plots saved!")
print("  - training_loss_comparison.png")
print("  - trajectory_comparison_ham_vs_std.png")
print("  - energy_conservation_comparison.png")
print("  - energy_error_comparison.png")
print("  - relative_energy_error_comparison.png")
print("  - phase_space_comparison.png")
