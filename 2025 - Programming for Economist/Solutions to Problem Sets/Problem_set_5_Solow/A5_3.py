# Savings-rate step experiment: s increases by 20% at t_switch.
# Plots: k_t (per effective worker) and c_t (per effective worker).

import numpy as np
import matplotlib.pyplot as plt
from A5_1 import simulate_solow_gn

# If your simulator is in a separate file:
# from solow_sim_gn import simulate_solow_gn
# Otherwise, assume simulate_solow_gn is already defined in the environment.

def make_step_s(T: int, s0: float, t_switch: int, up_factor: float = 1.20) -> np.ndarray:
    """
    Build a path for s_t of length T+1:
      s_t = s0            for t < t_switch
      s_t = min(up*s0, .99) for t >= t_switch  (avoid s>=1)
    """
    s_path = np.full(T+1, s0, dtype=float)
    if 0 <= t_switch <= T:
        s1 = min(up_factor * s0, 0.99)
        s_path[t_switch:] = s1
    return s_path

# ------------------ Parameters ------------------
T        = 80
t_switch = 20
alpha    = 1/3
s0       = 0.20
delta    = 0.08
g        = 0.02    # tech growth
n        = 0.01    # population growth
K0 = A0 = L0 = 1.0

# ------------------ Baseline (constant s = s0) ------------------
base = simulate_solow_gn(
    K0=K0, A0=A0, L0=L0, T=T,
    alpha=alpha, s=s0, delta=delta, g=g, n=n
)

# ------------------ Policy: s rises by 20% at t_switch -------------
s_path = make_step_s(T, s0, t_switch, up_factor=1.20)
policy = simulate_solow_gn(
    K0=K0, A0=A0, L0=L0, T=T,
    alpha=alpha, s=s_path, delta=delta, g=g, n=n
)

# ------------------ Series per effective worker -------------------
tgrid   = np.arange(T+1)
k_base  = base["k_eff"]
k_pol   = policy["k_eff"]
c_base  = base["C"] / (base["A"] * base["L"])
c_pol   = policy["C"] / (policy["A"] * policy["L"])

# ------------------ Plots -----------------------------------------
# (1) Capital per effective worker
plt.figure(figsize=(7,4))
plt.plot(tgrid, k_base, label="Baseline (constant s)")
plt.plot(tgrid, k_pol,  label="s ↑ 20% at t = {}".format(t_switch))
plt.axvline(t_switch, linestyle="--", alpha=0.7, label="policy switch")
plt.title("Capital per effective worker, $k_t$")
plt.xlabel("t")
plt.ylabel("$k_t$")
plt.legend(loc="best")
plt.tight_layout()
plt.show()

# (2) Consumption per effective worker
plt.figure(figsize=(7,4))
plt.plot(tgrid, c_base, label="Baseline (constant s)")
plt.plot(tgrid, c_pol,  label="s ↑ 20% at t = {}".format(t_switch))
plt.axvline(t_switch, linestyle="--", alpha=0.7, label="policy switch")
plt.title("Consumption per effective worker, $c_t$")
plt.xlabel("t")
plt.ylabel("$c_t$")
plt.legend(loc="best")
plt.tight_layout()
plt.show()
