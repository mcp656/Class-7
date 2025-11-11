import numpy as np
import matplotlib.pyplot as plt
from A1 import f, k_star_no_growth
from A2_1 import k_star_with_g
from A3_1 import k_star_gn

# --- parameters (edit as desired) -------------------------------------------
alpha = 1/3
s     = 0.20
delta = 0.08
g     = 0.02
n     = 0.01

# --- compute steady states and choose common grid ---------------------------
k_ng  = k_star_no_growth(alpha, s, delta)
k_g   = k_star_with_g(alpha, s, delta, g)
k_gn  = k_star_gn(alpha, s, delta, g, n)

kmax  = max(4.0, 2.5*max(k_ng, k_g, k_gn))
kgrid = np.linspace(1e-6, kmax, 600)
invest = s * f(kgrid, alpha)  # same across panels

# --- plotting ---------------------------------------------------------------
fig, axes = plt.subplots(3, 1, figsize=(6, 9), sharex=True)

# (1) no growth
axes[0].plot(kgrid, invest, label=r"$s\,f(k)$")
axes[0].plot(kgrid, delta * kgrid, label=r"$\delta\,k$")
axes[0].axvline(k_ng, ls="--", label=fr"$k^*={k_ng:.3f}$")
axes[0].set_title("Solow — no growth")
axes[0].set_ylabel("investment / break-even")
axes[0].legend(loc="best")

# (2) tech only
axes[1].plot(kgrid, invest, label=r"$s\,f(k)$")
axes[1].plot(kgrid, (delta + g) * kgrid, label=r"$(\delta+g)\,k$")
axes[1].axvline(k_g, ls="--", label=fr"$k^*={k_g:.3f}$")
axes[1].set_title("Solow — tech growth")
axes[1].set_ylabel("investment / break-even")
axes[1].legend(loc="best")

# (3) tech + population
be_gn = ((1 + g)*(1 + n) - (1 - delta)) * kgrid
axes[2].plot(kgrid, invest, label=r"$s\,f(k)$")
axes[2].plot(kgrid, be_gn, label=r"$[(1+g)(1+n)-(1-\delta)]\,k$")
axes[2].axvline(k_gn, ls="--", label=fr"$k^*={k_gn:.3f}$")
axes[2].set_title("Solow — tech + pop growth")
axes[2].set_xlabel(r"$k$ (capital per effective worker)")
axes[2].set_ylabel("investment / break-even")
axes[2].legend(loc="best")

fig.tight_layout()
plt.show()

print(f"k* no growth = {k_ng:.6f},  k* tech = {k_g:.6f},  k* tech+pop = {k_gn:.6f}")
