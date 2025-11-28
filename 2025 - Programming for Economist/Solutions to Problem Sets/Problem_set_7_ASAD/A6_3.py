# SOLUTION 6.3 — Tiny grid search over (a1, a2) + heatmap
import numpy as np
import matplotlib.pyplot as plt
from A6_1 import make_shocks_ps
from A6_2 import simulate_with_rule_ps, loss_quad_ps
from functions import par

def tiny_grid_search_ps(p, a1_vals, a2_vals, T=150, lam=0.5, seed=1, pad=0.6, n=400):
    """
    For each (a1, a2) on a coarse grid, simulate the model with fixed shocks
    and compute the quadratic loss. Return the loss matrix and the best pair.
    """
    # fixed shocks for the whole search
    v, s = make_shocks_ps(T, p, seed=seed)

    L = np.empty((len(a1_vals), len(a2_vals)))

    for i, a1 in enumerate(a1_vals):
        for j, a2 in enumerate(a2_vals):
            y, pi = simulate_with_rule_ps(a1, a2, v, s, p, pad=pad, n=n)
            L[i, j] = loss_quad_ps(y, pi, p, lam)

    # locate minimum
    i0, j0 = np.unravel_index(np.argmin(L), L.shape)
    best = {
        "a1": float(a1_vals[i0]),
        "a2": float(a2_vals[j0]),
        "loss": float(L[i0, j0]),
    }

    # heatmap of losses
    plt.figure(figsize=(6, 4))
    plt.imshow(
        L.T,
        origin="lower",
        aspect="auto",
        extent=[a1_vals[0], a1_vals[-1], a2_vals[0], a2_vals[-1]],
    )
    plt.colorbar(label="loss")
    plt.scatter(best["a1"], best["a2"], c="w", edgecolor="k", s=80)
    plt.xlabel("a1 (inflation response)")
    plt.ylabel("a2 (output-gap response)")
    plt.title(f"Best: a1={best['a1']:.2f}, a2={best['a2']:.2f}, loss={best['loss']:.3g}")
    plt.tight_layout()
    plt.show()

    return L, best


# example parameters for dynamics
p_dyn = {
    **par,
    "delta":   0.80,
    "omega":   0.15,
    "sigma_x": 0.01,
    "sigma_c": 0.005,
}

a1_vals = np.linspace(1.1, 2.2, 9)
a2_vals = np.linspace(0.00, 0.40, 9)

L, best = tiny_grid_search_ps(p_dyn, a1_vals, a2_vals, T=150, lam=0.5, seed=7)
print(best)
