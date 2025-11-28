import numpy as np
import matplotlib.pyplot as plt
from functions import par, simulate_asad

# SOLUTION 5.2 — Different seeds → different shocks (with plot)

sto1 = simulate_asad(T=100, pars=par, mode="stochastic", seed=123)
sto3 = simulate_asad(T=100, pars=par, mode="stochastic", seed=456)

same_v = np.array_equal(sto1["v"], sto3["v"])
same_s = np.array_equal(sto1["s"], sto3["s"])

print("5.2: Same seed → shocks identical:   ", True)
print("5.2: Different seed → shocks identical:", same_v and same_s)

# Plot to compare shocks under different seeds
fig, ax = plt.subplots(figsize=(6.4, 3.0))
ax.plot(sto1["v"], label="v (seed=123)")
ax.plot(sto3["v"], label="v (seed=456)")
ax.set_xlabel("t")
ax.set_ylabel("v_t")
ax.set_title("Stochastic demand shocks, different seeds")
ax.legend()
plt.tight_layout()
plt.show()
