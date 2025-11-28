# SOLUTION 6.3 — plot and compare stochastic paths

import numpy as np
import matplotlib.pyplot as plt
from A8_1 import par_supply, res_fix_sto, res_flex_sto  
from functions import plot_irfs_compare

# Reuse the helper from the notebook (if available)
plot_irfs_compare(
    res_fix_sto,
    res_flex_sto,
    title_suffix="(stochastic, supply-dominated shocks)",
)


# Option B: custom plots 
# t = np.arange(T)
# fig, axes = plt.subplots(2, 1, figsize=(6.4, 4.8), sharex=True)
#
# axes[0].plot(t, res_fix_sto["pi"],  label="fixed")
# axes[0].plot(t, res_flex_sto["pi"], label="flex")
# axes[0].axhline(pi_star, ls="--", c="k", lw=1)
# axes[0].set_ylabel("pi_t")
# axes[0].set_title("Inflation paths (supply-dominated shocks)")
# axes[0].legend()
#
# axes[1].plot(t, res_fix_sto["y"],  label="fixed")
# axes[1].plot(t, res_flex_sto["y"], label="flex")
# axes[1].axhline(ybar, ls="--", c="k", lw=1)
# axes[1].set_xlabel("t")
# axes[1].set_ylabel("y_t")
# axes[1].set_title("Output paths (supply-dominated shocks)")
# axes[1].legend()
#
# plt.tight_layout()
# plt.show()