#!/usr/bin/env python3
"""
Fast, reliable Option-A experiment
Standard vs Hamiltonian + Velocity Verlet
Conservative + Damped controls
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import json

np.random.seed(42)
torch.manual_seed(42)

G, L, M, DT = 9.81, 1.0, 1.0, 0.01
T_MAX = 8.0
STEPS = int(T_MAX / DT)
N_TRAJ = 60
N_EPOCHS = 25
BATCH = 256

OUT = Path(__file__).parent / "results_fast"
OUT.mkdir(exist_ok=True)

def pendulum_deriv(state, damping=0.0):
    th, om = state
    return np.array([om, -(G/L)*np.sin(th) - damping*om])

def rk4_step(state, dt, damping=0.0):
    k1 = pendulum_deriv(state, damping)
    k2 = pendulum_deriv(state + 0.5*dt*k1, damping)
    k3 = pendulum_deriv(state + 0.5*dt*k2, damping)
    k4 = pendulum_deriv(state + dt*k3, damping)
    return state + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)

def simulate(th0, om0, damping=0.0):
    s = np.zeros((STEPS, 2))
    s[0] = [th0, om0]
    for i in range(1, STEPS):
        s[i] = rk4_step(s[i-1], DT, damping)
    return s

def energy(traj):
    th, om = traj[:,0], traj[:,1]
    return 0.5*M*(L*om)**2 + M*G*L*(1 - np.cos(th))

class StdNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2,48), nn.Tanh(),
                                 nn.Linear(48,48), nn.Tanh(),
                                 nn.Linear(48,2))
    def forward(self, x): return self.net(x)

class HamNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2,48), nn.Tanh(),
                                 nn.Linear(48,48), nn.Tanh(),
                                 nn.Linear(48,1))
    def forward(self, qp): return self.net(qp)

    def grad_H(self, qp, create_graph=False):
        qp = qp.detach().clone().requires_grad_(True)
        H = self.forward(qp).sum()
        dH = torch.autograd.grad(H, qp, create_graph=create_graph)[0]
        return dH[:,0:1], dH[:,1:2]

    def verlet(self, qp, create_graph=False):
        q, p = qp[:,0:1], qp[:,1:2]
        dHdq, _ = self.grad_H(qp, create_graph)
        p_h = p - 0.5*DT*dHdq
        _, dHdp = self.grad_H(torch.cat([q, p_h],1), create_graph)
        q_n = q + DT*dHdp
        dHdq2, _ = self.grad_H(torch.cat([q_n, p_h],1), create_graph)
        p_n = p_h - 0.5*DT*dHdq2
        out = torch.cat([q_n, p_n],1)
        return out if create_graph else out.detach()

def make_data(damping):
    X, Y = [], []
    for _ in range(N_TRAJ):
        th0 = np.random.uniform(-1.2, 1.2)
        om0 = np.random.uniform(-0.8, 0.8)
        s = simulate(th0, om0, damping)
        X.append(s[:-1]); Y.append(s[1:])
    return np.vstack(X).astype(np.float32), np.vstack(Y).astype(np.float32)

def train(model, X, Y, name, is_ham=False):
    Xt, Yt = torch.from_numpy(X), torch.from_numpy(Y)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.MSELoss()
    losses = []
    n = len(Xt)
    for ep in range(N_EPOCHS):
        perm = torch.randperm(n)
        eloss = 0.0
        for i in range(0, n, BATCH):
            idx = perm[i:i+BATCH]
            xb, yb = Xt[idx], Yt[idx]
            opt.zero_grad()
            pred = model.verlet(xb, create_graph=True) if is_ham else model(xb)
            loss = crit(pred, yb)
            loss.backward()
            opt.step()
            eloss += loss.item() * len(xb)
        eloss /= n
        losses.append(eloss)
        if (ep+1) % 5 == 0 or ep == 0:
            print(f"  [{name}] ep {ep+1:2d} loss {eloss:.7f}", flush=True)
    return losses

def roll_std(model, th0, om0):
    st = torch.tensor([[th0, om0]], dtype=torch.float32)
    tr = [st.numpy().squeeze()]
    model.eval()
    with torch.no_grad():
        for _ in range(STEPS-1):
            st = model(st)
            tr.append(st.numpy().squeeze())
    return np.array(tr)

def roll_ham(model, th0, om0):
    st = torch.tensor([[th0, om0]], dtype=torch.float32)
    tr = [st.numpy().squeeze()]
    model.eval()
    for _ in range(STEPS-1):
        with torch.enable_grad():
            st = model.verlet(st.detach(), create_graph=False)
        tr.append(st.detach().numpy().squeeze())
    return np.array(tr)

def report(trueE, predE, label):
    err = np.abs(predE - trueE)
    rel = err / (abs(trueE[0]) + 1e-12)
    print(f"{label}: mean_rel={np.mean(rel)*100:.4f}%  max_rel={np.max(rel)*100:.4f}%", flush=True)
    return {"mean_rel_pct": float(np.mean(rel)*100), "max_rel_pct": float(np.max(rel)*100),
            "mean_err": float(np.mean(err)), "max_err": float(np.max(err))}

def run(damping, tag):
    print(f"\n=== {tag.upper()} (damping={damping}) ===", flush=True)
    X, Y = make_data(damping)
    print(f"Data {X.shape}", flush=True)

    std, ham = StdNet(), HamNet()
    print("Train Standard", flush=True)
    sl = train(std, X, Y, "Std", is_ham=False)
    print("Train Hamiltonian", flush=True)
    hl = train(ham, X, Y, "Ham", is_ham=True)

    th0, om0 = np.radians(55), 0.0
    true = simulate(th0, om0, damping)
    s_tr = roll_std(std, th0, om0)
    h_tr = roll_ham(ham, th0, om0)

    tE, sE, hE = energy(true), energy(s_tr), energy(h_tr)
    m = {}
    m["standard"] = report(tE, sE, "Standard")
    m["hamiltonian"] = report(tE, hE, "Hamiltonian")

    t = np.linspace(0, T_MAX, STEPS)
    for name, data in [("loss", (sl, hl)),
                       ("traj", (np.degrees(true[:,0]), np.degrees(h_tr[:,0]), np.degrees(s_tr[:,0]))),
                       ("energy", (tE, hE, sE))]:
        fig, ax = plt.subplots(figsize=(9,4))
        if name == "loss":
            ax.semilogy(data[0], 'r', label='Std'); ax.semilogy(data[1], 'g', label='Ham')
        elif name == "traj":
            ax.plot(t, data[0], 'k', lw=2, label='True'); ax.plot(t, data[1], 'g--', label='Ham'); ax.plot(t, data[2], 'r--', label='Std')
        else:
            ax.plot(t, data[0], 'k', lw=2, label='True'); ax.plot(t, data[1], 'g--', label='Ham'); ax.plot(t, data[2], 'r--', label='Std')
        ax.set_title(f"{name} \u2014 {tag}"); ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(OUT / f"{name}_{tag}.png", dpi=120); plt.close()

    fig, axes = plt.subplots(1,3, figsize=(11,3.5))
    for ax, tr, tit, c in zip(axes, [true, h_tr, s_tr], ['True','Ham','Std'], ['k','g','r']):
        ax.plot(tr[:,0], tr[:,1], c=c, lw=1.2)
        ax.set_title(tit); ax.grid(alpha=0.3)
    fig.suptitle(f"Phase \u2014 {tag}"); fig.tight_layout()
    fig.savefig(OUT / f"phase_{tag}.png", dpi=120); plt.close()
    print(f"Plots \u2192 {OUT}", flush=True)
    return m

if __name__ == "__main__":
    print("Fast Option-A start", flush=True)
    mc = run(0.0, "conservative")
    md = run(0.15, "damped")
    summary = {"conservative": mc, "damped": md}
    with open(OUT / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\n===== FINAL TABLE =====", flush=True)
    print(f"{'Model':<15} {'Cons %':>10} {'Damped %':>10}", flush=True)
    for k in ["standard", "hamiltonian"]:
        print(f"{k:<15} {mc[k]['mean_rel_pct']:10.4f} {md[k]['mean_rel_pct']:10.4f}", flush=True)
    print("DONE", flush=True)
