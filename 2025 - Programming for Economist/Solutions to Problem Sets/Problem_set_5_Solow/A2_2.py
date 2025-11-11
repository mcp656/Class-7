import numpy as np
import matplotlib.pyplot as plt
from A2_1 import k_star_with_g
from A1 import f

# --- Parameters (edit here) -----------------------------------------
alpha = 1/3
s     = 0.20
delta = 0.08
g     = 0.02   # technology growth; population growth n=0 here

# --- Steady state and grid --------------------------------------------------
k_star = k_star_with_g(alpha, s, delta, g)
kmax   = max(4.0, 2.5*k_star)
k      = np.linspace(1e-6, kmax, 600)

invest     = s * f(k, alpha)           # s f(k)
break_even = (delta + g) * k           # (δ+g)k

# --- Plot -------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(k, invest, label=r"$s\,f(k)$")
ax.plot(k, break_even, label=r"$(\delta+g)\,k$")
ax.axvline(k_star, linestyle="--", color="black", alpha=0.7, label=fr"$k^*={k_star:.3f}$")
ax.scatter([k_star], [s * f(k_star, alpha)], zorder=3)  # mark intersection

ax.set_title("Solow steady state with technology growth ($n=0$)")
ax.set_xlabel(r"$k$ (capital per effective worker)")
ax.set_ylabel(r"investment / break-even")
ax.set_xlim(0, kmax)
ax.set_ylim(0, 1.05*max(invest.max(), break_even.max()))
ax.legend(loc="best")
fig.tight_layout()
plt.show()

print(f"Parameters: alpha={alpha:.3f}, s={s:.3f}, delta={delta:.3f}, g={g:.3f}")
print(f"Steady state: k* = {k_star:.6f}")