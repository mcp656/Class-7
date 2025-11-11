import numpy as np
import matplotlib.pyplot as plt
from A1 import f
from A3_1 import k_star_gn

plt.rcParams["text.usetex"] = False  # keep mathtext simple/portable

# --- Parameters --------------------------------------------------------------
alpha = 1/3
s     = 0.20
delta = 0.08
g     = 0.02   # keep g>0
n     = 0.01   # n>0

# --- Compute steady state and curves ----------------------------------------
k_star = k_star_gn(alpha, s, delta, g, n)
kmax   = max(4.0, 2.5 * k_star)
k      = np.linspace(1e-6, kmax, 600)

invest     = s * f(k, alpha)                         # s f(k)
break_even = ((1 + g) * (1 + n) - (1 - delta)) * k   # (δ+g+n+gn) k

# --- Plot -------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(k, invest, label=r"$s\,f(k)$")
ax.plot(k, break_even, label=r"$[(\delta + g + n + gn )]\,k$")
ax.axvline(k_star, linestyle="--", color="black", alpha=0.7, label=fr"$k^*={k_star:.3f}$")
ax.scatter([k_star], [s * f(k_star, alpha)], zorder=3)

ax.set_title("Solow steady state with tech (g>0) and population growth (n>0)")
ax.set_xlabel(r"$k$ (capital per effective worker)")
ax.set_ylabel("investment / break-even")
ax.set_xlim(0, kmax)
ax.set_ylim(0, 1.05 * max(invest.max(), break_even.max()))
ax.legend(loc="best")
fig.tight_layout()
plt.show()

print(f"Parameters: alpha={alpha:.3f}, s={s:.3f}, delta={delta:.3f}, g={g:.3f}, n={n:.3f}")
print(f"Steady state: k* = {k_star:.6f}")
