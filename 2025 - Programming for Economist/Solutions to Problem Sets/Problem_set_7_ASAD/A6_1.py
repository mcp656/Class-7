# SOLUTION 6.1 — AR(1) shocks for v_t and s_t

import numpy as np
import matplotlib.pyplot as plt
from functions import par
# dynamic parameter dict for the policy search
p_dyn = {
    **par,           # copy all baseline parameters
    "delta":   0.80, # persistence of demand shock v_t
    "omega":   0.15, # persistence of supply shock s_t
    "sigma_x": 0.01, # std dev of v_t innovation
    "sigma_c": 0.005 # std dev of s_t innovation
}

a1_vals = np.linspace(1.1, 2.2, 9)
a2_vals = np.linspace(0.00, 0.40, 9)

def make_shocks_ps(T, p, seed=0):
    """
    Build AR(1) processes for demand shocks v_t and supply shocks s_t.

    v_t = delta * v_{t-1} + x_t
    s_t = omega * s_{t-1} + c_t
    """
    rng = np.random.default_rng(seed)

    x = rng.normal(0.0, p["sigma_x"], size=T)
    c = rng.normal(0.0, p["sigma_c"], size=T)

    v = np.zeros(T)
    s = np.zeros(T)

    for t in range(1, T):
        v[t] = p["delta"] * v[t-1] + x[t]
        s[t] = p["omega"] * s[t-1] + c[t]

    return v, s

# Example (optional): visualize one realization
v_example, s_example = make_shocks_ps(T=150, p=p_dyn, seed=1)
plt.plot(v_example, label="v_t")
plt.plot(s_example, label="s_t")
plt.legend(); plt.tight_layout(); plt.show()
