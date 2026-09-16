import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for CI/CD

np.random.seed(42)
torch.manual_seed(42)

# Physical constants
g = 9.81
L = 1.0

def pendulum_deriv(state, t):
    theta, omega = state
    dtheta_dt = omega
    domega_dt = -(g / L) * np.sin(theta)
    return np.array([dtheta_dt, domega_dt])

def simulate(theta0, omega0, t_max=10.0, dt=0.01):
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

def compute_energy(trajectory, g=9.81, L=1.0):
    theta = trajectory[:, 0]
    omega = trajectory[:, 1]
    kinetic = 0.5 * omega**2
    potential = (g / L) * (1 - np.cos(theta))
    return kinetic + potential

def generate_dataset(n_trajectories=200, t_max=10.0, dt=0.01):
    X, Y = [], []
    for _ in range(n_trajectories):
        theta0 = np.random.uniform(-np.pi/2, np.pi/2)
        omega0 = np.random.uniform(-1, 1)
        states = simulate(theta0, omega0, t_max, dt)
        X.append(states[:-1])
        Y.append(states[1:])
    return np.vstack(X), np.vstack(Y)

class StandardNet(nn.Module):
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

class HamiltonianNet(nn.Module):
    def __init__(self, hidden_size=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, q):
        return self.net(q)
    
    def hamiltonian_dynamics(self, q, dt=0.01):
        q_copy = q.clone().detach().requires_grad_(True)
        H = self.forward(q_copy)
        H_sum = H.sum()
        H_sum.backward()
        dH_dq = q_copy.grad
        dtheta_dt = dH_dq[:, 1:2]
        dp_dt = -dH_dq[:, 0:1]
        dq_dt = torch.cat([dtheta_dt, dp_dt], dim=1)
        q_next = q.detach() + dt * dq_dt.detach()
        return q_next

print("Generating training data...")
X, Y = generate_dataset()
print(f"Dataset shape: {X.shape}, {Y.shape}\n")

X_tensor = torch.tensor(X, dtype=torch.float32)
Y_tensor = torch.tensor(Y, dtype=torch.float32)

print("Initializing networks...")
std_net = StandardNet(hidden_size=64)
ham_net = HamiltonianNet(hidden_size=64)

criterion = nn.MSELoss()
std_optimizer = optim.Adam(std_net.parameters(), lr=0.001)
ham_optimizer = optim.Adam(ham_net.parameters(), lr=0.001)

n_epochs = 50
batch_size = 256
n_samples = len(X_tensor)

std_losses, ham_losses = [], []

print("Training networks...\n")
for epoch in range(n_epochs):
    perm = torch.randperm(n_samples)
    std_loss_sum, ham_loss_sum = 0.0, 0.0
    
    for i in range(0, n_samples, batch_size):
        idx = perm[i:i+batch_size]
        x_batch = X_tensor[idx]
        y_batch = Y_tensor[idx]
        
        std_optimizer.zero_grad()
        std_pred = std_net(x_batch)
        std_loss = criterion(std_pred, y_batch)
        std_loss.backward()
        std_optimizer.step()
        std_loss_sum += std_loss.item() * len(x_batch)
        
        ham_optimizer.zero_grad()
        ham_pred = ham_net.hamiltonian_dynamics(x_batch, dt=0.01)
        ham_loss = criterion(ham_pred, y_batch)
        ham_loss.backward()
        ham_optimizer.step()
        ham_loss_sum += ham_loss.item() * len(x_batch)
    
    std_avg_loss = std_loss_sum / n_samples
    ham_avg_loss = ham_loss_sum / n_samples
    std_losses.append(std_avg_loss)
    ham_losses.append(ham_avg_loss)
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:2d}: Std Loss = {std_avg_loss:.8f}, Ham Loss = {ham_avg_loss:.8f}")

print("\nTraining complete!")

def rollout_standard(model, theta0, omega0, steps=999, dt=0.01):
    state = torch.tensor([theta0, omega0], dtype=torch.float32)
    trajectory = [state.numpy()]
    with torch.no_grad():
        for _ in range(steps):
            state = model(state.unsqueeze(0)).squeeze(0)
            trajectory.append(state.numpy())
    return np.array(trajectory)

def rollout_hamiltonian(model, theta0, omega0, steps=999, dt=0.01):
    state = torch.tensor([theta0, omega0], dtype=torch.float32).unsqueeze(0)
    trajectory = [state.numpy().squeeze()]
    with torch.no_grad():
        for _ in range(steps):
            state = model.hamiltonian_dynamics(state, dt=dt)
            trajectory.append(state.numpy().squeeze())
    return np.array(trajectory)

theta0_test = np.radians(60)
omega0_test = 0.0

true_traj = simulate(theta0_test, omega0_test)
std_traj = rollout_standard(std_net, theta0_test, omega0_test)
ham_traj = rollout_hamiltonian(ham_net, theta0_test, omega0_test)

true_energy = compute_energy(true_traj)
std_energy = compute_energy(std_traj)
ham_energy = compute_energy(ham_traj)

t = np.linspace(0, 10, len(true_traj))

print("\n" + "="*60)
print("ENERGY CONSERVATION ANALYSIS")
print("="*60)
E0 = true_energy[0]
std_error = np.abs(std_energy - true_energy)
ham_error = np.abs(ham_energy - true_energy)
print(f"\nStandard Network:")
print(f"  Initial energy: {E0:.6f}")
print(f"  Mean error: {np.mean(std_error):.8f}")
print(f"  Max error: {np.max(std_error):.8f}")
print(f"  Mean rel. error: {np.mean(std_error/E0):.4%}")
print(f"  Max rel. error: {np.max(std_error/E0):.4%}")

print(f"\nHamiltonian Network:")
print(f"  Initial energy: {E0:.6f}")
print(f"  Mean error: {np.mean(ham_error):.8f}")
print(f"  Max error: {np.max(ham_error):.8f}")
print(f"  Mean rel. error: {np.mean(ham_error/E0):.4%}")
print(f"  Max rel. error: {np.max(ham_error/E0):.4%}")

improvement = np.mean(std_error) / np.mean(ham_error)
print(f"\n→ Hamiltonian is {improvement:.1f}x better at conserving energy")
print("="*60 + "\n")

# Generate all visualizations
print("Generating visualizations...\n")

# 1. Training Loss Comparison
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(std_losses, label='Standard Network', linewidth=2.5, color='red', alpha=0.8)
ax.plot(ham_losses, label='Hamiltonian Network', linewidth=2.5, color='green', alpha=0.8)
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Training Loss (MSE)', fontsize=12)
ax.set_title('Training Loss Comparison', fontsize=13, fontweight='bold')
ax.set_yscale('log')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, which='both')
plt.tight_layout()
plt.savefig('training_loss_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ training_loss_comparison.png")

# 2. Trajectory Comparison
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(t, np.degrees(true_traj[:, 0]), label='True physics (RK4)', linewidth=2.5, color='black')
ax.plot(t, np.degrees(ham_traj[:, 0]), label='Hamiltonian NN', linestyle='--', linewidth=2, color='green', alpha=0.8)
ax.plot(t, np.degrees(std_traj[:, 0]), label='Standard NN', linestyle='--', linewidth=2, color='red', alpha=0.8)
ax.set_xlabel('Time (s)', fontsize=12)
ax.set_ylabel('Angle (degrees)', fontsize=12)
ax.set_title('Pendulum Angle: Hamiltonian vs Standard Network', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('trajectory_comparison_ham_vs_std.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ trajectory_comparison_ham_vs_std.png")

# 3. Energy Conservation Comparison (THE KEY PLOT)
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(t, true_energy, label='True energy (conserved)', linewidth=2.5, color='black')
ax.plot(t, ham_energy, label='Hamiltonian NN energy', linestyle='--', linewidth=2, color='green', alpha=0.8)
ax.plot(t, std_energy, label='Standard NN energy', linestyle='--', linewidth=2, color='red', alpha=0.8)
ax.set_xlabel('Time (s)', fontsize=12)
ax.set_ylabel('Total Energy (normalized)', fontsize=12)
ax.set_title('Energy Conservation: Hamiltonian vs Standard Network', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('energy_conservation_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ energy_conservation_comparison.png")

# 4. Absolute Energy Error
fig, ax = plt.subplots(figsize=(14, 6))
ax.semilogy(t, np.abs(ham_energy - true_energy), label='Hamiltonian NN', linewidth=2.5, color='green')
ax.semilogy(t, np.abs(std_energy - true_energy), label='Standard NN', linewidth=2.5, color='red')
ax.set_xlabel('Time (s)', fontsize=12)
ax.set_ylabel('Absolute Energy Error (log scale)', fontsize=12)
ax.set_title('Energy Error Accumulation Over Time', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, which='both', alpha=0.3)
plt.tight_layout()
plt.savefig('energy_error_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ energy_error_comparison.png")

# 5. Relative Energy Error
fig, ax = plt.subplots(figsize=(14, 6))
ax.semilogy(t, np.abs(ham_energy - true_energy) / np.abs(true_energy[0]), 
            label='Hamiltonian NN', linewidth=2.5, color='green')
ax.semilogy(t, np.abs(std_energy - true_energy) / np.abs(true_energy[0]), 
            label='Standard NN', linewidth=2.5, color='red')
ax.set_xlabel('Time (s)', fontsize=12)
ax.set_ylabel('Relative Energy Error (log scale)', fontsize=12)
ax.set_title('Relative Energy Error Over Time', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, which='both', alpha=0.3)
plt.tight_layout()
plt.savefig('relative_energy_error_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ relative_energy_error_comparison.png")

# 6. Phase Space Comparison
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

axes[0].plot(true_traj[:, 0], true_traj[:, 1], linewidth=2.5, color='black')
axes[0].scatter(true_traj[0, 0], true_traj[0, 1], s=150, marker='o', color='green', 
                edgecolor='black', linewidth=2, label='Start', zorder=5)
axes[0].scatter(true_traj[-1, 0], true_traj[-1, 1], s=150, marker='X', color='red', 
                edgecolor='black', linewidth=2, label='End', zorder=5)
axes[0].set_xlabel('Angle (rad)', fontsize=11)
axes[0].set_ylabel('Angular velocity (rad/s)', fontsize=11)
axes[0].set_title('True Physics (RK4)', fontsize=12, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

axes[1].plot(ham_traj[:, 0], ham_traj[:, 1], linewidth=2.5, color='green')
axes[1].scatter(ham_traj[0, 0], ham_traj[0, 1], s=150, marker='o', color='green', 
                edgecolor='black', linewidth=2, label='Start', zorder=5)
axes[1].scatter(ham_traj[-1, 0], ham_traj[-1, 1], s=150, marker='X', color='red', 
                edgecolor='black', linewidth=2, label='End', zorder=5)
axes[1].set_xlabel('Angle (rad)', fontsize=11)
axes[1].set_ylabel('Angular velocity (rad/s)', fontsize=11)
axes[1].set_title('Hamiltonian Network', fontsize=12, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

axes[2].plot(std_traj[:, 0], std_traj[:, 1], linewidth=2.5, color='red')
axes[2].scatter(std_traj[0, 0], std_traj[0, 1], s=150, marker='o', color='green', 
                edgecolor='black', linewidth=2, label='Start', zorder=5)
axes[2].scatter(std_traj[-1, 0], std_traj[-1, 1], s=150, marker='X', color='red', 
                edgecolor='black', linewidth=2, label='End', zorder=5)
axes[2].set_xlabel('Angle (rad)', fontsize=11)
axes[2].set_ylabel('Angular velocity (rad/s)', fontsize=11)
axes[2].set_title('Standard Network', fontsize=12, fontweight='bold')
axes[2].legend(fontsize=10)
axes[2].grid(True, alpha=0.3)

plt.suptitle('Phase Space Trajectories (θ vs ω)', fontsize=13, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig('phase_space_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ phase_space_comparison.png")

print("\n" + "="*60)
print("All visualizations generated successfully!")
print("="*60)
