# Extended Results (Velocity Verlet + Negative Control)

## Summary

| Model | Conservative (mean rel. energy error) | Damped (mean rel. energy error) |
|-------|---------------------------------------|---------------------------------|
| Standard Network | 26.54 % | 37.09 % |
| Hamiltonian + Verlet | **0.22 %** | 40.92 % |

### Key findings

- On the **conservative** pendulum the Hamiltonian network with velocity Verlet reduces residual energy error by ~120×.
- On the **damped** pendulum the same inductive bias loses its advantage (and slightly underperforms). This is the required negative control.

## Files added

- `experiment_fast.py` — clean, reproducible experiment (Standard vs Hamiltonian, Verlet integrator, both controls)
- `results_fast/metrics.json` — exact numerical results
- `paper.tex` — full research manuscript

## How to reproduce

```bash
pip install -r requirements.txt
python experiment_fast.py
```

The script generates plots and writes `results_fast/metrics.json`.
