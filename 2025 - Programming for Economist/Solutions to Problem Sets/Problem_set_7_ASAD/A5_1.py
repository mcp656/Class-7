
# SOLUTION 5.1 — Same seed → identical shocks and paths (with plots)

import numpy as np
import matplotlib.pyplot as plt
from functions import par, simulate_asad

sto1 = simulate_asad(T=100, pars=par, mode="stochastic", seed=123)
sto2 = simulate_asad(T=100, pars=par, mode="stochastic", seed=123)

# same shocks
assert np.array_equal(sto1["v"], sto2["v"])
assert np.array_equal(sto1["s"], sto2["s"])

# same simulated paths
assert np.array_equal(sto1["y"],  sto2["y"])
assert np.array_equal(sto1["pi"], sto2["pi"])
print("5.1: Same seed → shocks and paths are identical.")

# Plot to visualize that the paths overlap
fig, axes = plt.subplots(2, 1, figsize=(6.4, 4.8), sharex=True)

axes[0].plot(sto1["v"], label="v (run 1)")
axes[0].plot(sto2["v"], ls="--", label="v (run 2)")
axes[0].set_ylabel("v_t")
axes[0].set_title("Stochastic demand shocks, same seed")
axes[0].legend()

axes[1].plot(sto1["y"], label="y (run 1)")
axes[1].plot(sto2["y"], ls="--", label="y (run 2)")
axes[1].set_xlabel("t")
axes[1].set_ylabel("y_t")
axes[1].set_title("Output paths, same seed")
axes[1].legend()

plt.tight_layout()
plt.show()
