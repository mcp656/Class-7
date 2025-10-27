# --- CES consumer: price shock sensitivity (p1 doubles) ---
# Utility: u(x1,x2) = (alpha * x1^(-beta) + (1 - alpha) * x2^(-beta))^(-1/beta)
# Elasticity of substitution: sigma = 1 / (1 + beta)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

# Parameters 
# Use these baseline parameters
alpha = 0.5
beta = 0.000001
I = 10
p1 = 1
p2 = 2

betas    = np.array([0.1, 0.3, 0.5, 0.7, 0.9])  # five beta values in (0,1)

# ----- helpers -----
def sigma_from_beta(beta):
    # elasticity of substitution
    return 1.0 / (1.0 + beta)

def ces_demands(I, p1, p2, alpha, beta):
    """
    Returns (x1_star, x2_star, u_star, s1, s2) using CES duality.

    Indirect utility: u_star = I / C(p),
    where C(p) = [ alpha^sigma * p1^(1 - sigma) + (1 - alpha)^sigma * p2^(1 - sigma) ]^(1 / (1 - sigma))

    Budget shares:
      s1 = alpha^sigma * p1^(1 - sigma) / denom
      s2 = 1 - s1
    """
    sigma = sigma_from_beta(beta)
    denom = (alpha**sigma) * (p1**(1.0 - sigma)) + ((1.0 - alpha)**sigma) * (p2**(1.0 - sigma))
    s1 = (alpha**sigma) * (p1**(1.0 - sigma)) / denom
    s2 = 1.0 - s1
    x1 = (I / p1) * s1
    x2 = (I / p2) * s2
    C  = denom**(1.0 / (1.0 - sigma))
    u  = I / C
    return x1, x2, u, s1, s2

# ----- baseline vs. shock (p1 doubles) -----
base = np.array([ces_demands(I, p1,     p2, alpha, b) for b in betas]).T
shck = np.array([ces_demands(I, 2.0*p1, p2, alpha, b) for b in betas]).T

x1_b, x2_b, u_b = base[0], base[1], base[2]
x1_s, x2_s, u_s = shck[0], shck[1], shck[2]

pct = lambda new, old: 100.0 * (new - old) / old
d_x1 = pct(x1_s, x1_b)
d_x2 = pct(x2_s, x2_b)
d_u  = pct(u_s,  u_b)

# ----- plots -----

# plots
fig, axs = plt.subplots(1, 2, figsize=(9, 4), sharex=True)

# Panel A: quantities
axs[0].plot(betas, d_x1, marker='o', label='% change in x1_star')
axs[0].plot(betas, d_x2, marker='o', label='% change in x2_star')
axs[0].set_title('Quantities after p1 doubles')
axs[0].set_xlabel('beta  (lower beta -> higher sigma)')
axs[0].set_ylabel('% change from baseline')
axs[0].axhline(0, linewidth=1)
axs[0].yaxis.set_major_formatter(PercentFormatter())
axs[0].set_xticks(betas.tolist())
axs[0].grid(True, alpha=0.3)
axs[0].legend(frameon=False)

# Panel B: utility
axs[1].plot(betas, d_u, marker='o', label='% change in u_star')
axs[1].set_title('Utility after p1 doubles')
axs[1].set_xlabel('beta')
axs[1].set_ylabel('% change')
axs[1].axhline(0, linewidth=1)
axs[1].yaxis.set_major_formatter(PercentFormatter())
axs[1].set_xticks(betas.tolist())
axs[1].grid(True, alpha=0.3)
axs[1].legend(frameon=False)

plt.tight_layout()
plt.show()