import numpy as np
import matplotlib.pyplot as plt 
from functions import par, solve_grid   

# SOLUTION
def plot_overlay_closed(par, v, s, title, pad=0.6, n=400):
    """
    Overlay baseline and one shock scenario in the closed-economy AS–AD diagram.
    """

    # baseline: no shocks
    y_base, pi_base, y_grid, pi_ad_base, pi_sras_base = solve_grid(
        pi_e=par["pi_star"], v=0.0, s=0.0, p=par, pad=pad, n=n
    )

    # scenario: either demand shock (v != 0) or supply shock (s != 0)
    y_scen, pi_scen, _, pi_ad_scen, pi_sras_scen = solve_grid(
        pi_e=par["pi_star"], v=v, s=s, p=par, pad=pad, n=n
    )

    fig, ax = plt.subplots(figsize=(6.4, 4.2))

    # baseline curves
    ax.plot(y_grid, pi_sras_base, label="SRAS (base)", color="#666666")
    ax.plot(y_grid, pi_ad_base,   label="AD (base)",   color="#666666", ls="--")

    # scenario curves
    ax.plot(y_grid, pi_sras_scen, label="SRAS (scenario)", color="#1f77b4")
    ax.plot(y_grid, pi_ad_scen,   label="AD (scenario)",   color="#1f77b4", ls="--")

    # long-run output
    ax.axvline(par["ybar"], color="k", ls=":", lw=1.0, label="LRAS")

    # mark equilibria
    ax.scatter([y_base], [pi_base], color="k", s=35, label="Eq (base)")
    ax.scatter([y_scen], [pi_scen], facecolor="#1f77b4", edgecolor="k", s=55, label="Eq (scenario)")

    ax.set_xlabel("y")
    ax.set_ylabel("pi")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.show()

    print(f"Baseline:  y={y_base:.4f}, pi={pi_base:.4f}")
    print(f"Scenario:  y={y_scen:.4f}, pi={pi_scen:.4f}")


# demand shock: v = +0.02, s = 0
plot_overlay_closed(par, v=0.02, s=0.0, title="Demand shock: v = 0.02")

# supply shock: v = 0, s = +0.1
plot_overlay_closed(par, v=0.00, s=0.1, title="Supply shock: s = 0.1")
