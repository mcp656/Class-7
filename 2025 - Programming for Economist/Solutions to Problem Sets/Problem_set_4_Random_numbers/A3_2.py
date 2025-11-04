import numpy as np
import matplotlib.pyplot as plt
from A3_1 import spell_lengths
from A2_2 import simulate_many_workers_shocks



T = 10_000
N = 20_000
s = 0.02   # separation probability (E->U)
f = 0.30   # job-finding probability (U->E)
s0 = 0
seed = 7

states, Udraws, u_rate = simulate_many_workers_shocks(T, N, s, f, s0, seed)

# 1) Compute spell samples (runs of 1 = U, runs of 0 = E)
U_spells = spell_lengths(states, value=1)
E_spells = spell_lengths(states, value=0)

# Guard against empty arrays (shouldn't happen in long sims)
if U_spells.size == 0 or E_spells.size == 0:
    raise ValueError("No completed spells detected. Increase T or check simulation.")

# 2) Sample means vs geometric predictions
mean_U_sample = U_spells.mean()
mean_E_sample = E_spells.mean()
mean_U_theory = 1 / f
mean_E_theory = 1 / s

print(f"Unemployment spells (U=1): count={U_spells.size:,}")
print(f"  Sample mean length   : {mean_U_sample:.3f}")
print(f"  Theoretical 1/f      : {mean_U_theory:.3f}")
print()
print(f"Employment spells (E=0): count={E_spells.size:,}")
print(f"  Sample mean length   : {mean_E_sample:.3f}")
print(f"  Theoretical 1/s      : {mean_E_theory:.3f}")

# 3) Geometric pmf overlays
def geometric_pmf(k, p):
    """PMF for length k >= 1 with per-period exit prob p."""
    k = np.asarray(k)
    return (1 - p)**(k - 1) * p

# Choose x-ranges that cover most of the mass
kU_max = int(np.percentile(U_spells, 99.5))  # robust upper bound
kE_max = int(np.percentile(E_spells, 99.5))
kU = np.arange(1, max(2, kU_max + 1))
kE = np.arange(1, max(2, kE_max + 1))
pmf_U = geometric_pmf(kU, f)
pmf_E = geometric_pmf(kE, s)

# Controls
bin_w = 1.0      # bins (try 0.25 for even finer)
line_w = 1.0     # thickness of overlay line
ms = 3           # marker size

# 4) Plots: histogram (density) + pmf overlay
# U-spells
plt.figure()
plt.hist(U_spells, bins=np.arange(0.5, kU.max() + 1.5, bin_w), density=True)
plt.plot(kU, pmf_U, marker='o', linewidth=line_w, markersize=ms)
plt.xlabel("Unemployment spell length")
plt.ylabel("Density / Probability")
plt.title("U-spell lengths: histogram with geometric pmf overlay")
plt.show()

# E-spells
plt.figure()
plt.hist(E_spells, bins=np.arange(0.5, kE.max() + 1.5, bin_w), density=True)
plt.plot(kE, pmf_E, marker='o', linewidth=line_w, markersize=ms)
plt.xlabel("Employment spell length")
plt.ylabel("Density / Probability")
plt.title("E-spell lengths: histogram with geometric pmf overlay")
plt.show()
