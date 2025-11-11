# Simulate a **halving of population growth** and plot transitions of
# capital per effective worker (k_t) and consumption per effective worker (c_t).

import numpy as np
import matplotlib.pyplot as plt
from A5_1 import simulate_solow_gn

# -- assume simulate_solow_gn() is already defined (from your previous cell) --

def make_halved_path(T: int, n0: float, t_switch: int) -> np.ndarray:
    """
    Build a path for population growth n_t of length T+1:
      n_t = n0 for t < t_switch, and n0/2 for t >= t_switch
    """
    n_path = np.full(T+1, n0, dtype=float)
    if 0 <= t_switch <= T:
        n_path[t_switch:] = 0.5 * n0
    return n_path

# ------------------ Parameters ------------------
T       = 80
t_switch= 20
alpha   = 1/3
s       = 0.20
delta   = 0.08
g       = 0.02      # tech growth (kept constant)
n0      = 0.01      # baseline population growth

K0 = 1.0; A0 = 1.0; L0 = 1.0

# ------------------ Baseline (constant n = n0) ------------------
base = simulate_solow_gn(
    K0=K0, A0=A0, L0=L0, T=T,
    alpha=alpha, s=s, delta=delta,
    g=g, n=n0
)

# ------------------ Policy: n halves at t_switch ------------------
n_path_halved = make_halved_path(T, n0, t_switch)
half = simulate_solow_gn(
    K0=K0, A0=A0, L0=L0, T=T,
    alpha=alpha, s=s, delta=delta,
    g=g, n=n_path_halved
)

# Per-effective-worker series
k_base  = base["k_eff"]
k_half  = half["k_eff"]
c_base  = base["C"] / (base["A"] * base["L"])
c_half  = half["C"] / (half["A"] * half["L"])

# ------------------ Plots ------------------
tgrid = np.arange(T+1)

# (1) Capital per effective worker
plt.figure(figsize=(7,4))
plt.plot(tgrid, k_base,  label="Baseline (n = n₀)")
plt.plot(tgrid, k_half,  label="n halves at t = {}".format(t_switch))
plt.axvline(t_switch, linestyle="--", alpha=0.7, label="switch")
plt.title("Capital per effective worker, $k_t$")
plt.xlabel("t")
plt.ylabel("$k_t$")
plt.legend(loc="best")
plt.tight_layout()
plt.show()

# (2) Consumption per effective worker
plt.figure(figsize=(7,4))
plt.plot(tgrid, c_base, label="Baseline (n = n₀)")
plt.plot(tgrid, c_half, label="n halves at t = {}".format(t_switch))
plt.axvline(t_switch, linestyle="--", alpha=0.7, label="switch")
plt.title("Consumption per effective worker, $c_t$")
plt.xlabel("t")
plt.ylabel("$c_t$")
plt.legend(loc="best")
plt.tight_layout()
plt.show()
