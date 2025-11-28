import numpy as np
from functions import par, ad_curve, sras_curve, solve_grid 


# SOLUTION
def solve_grid_linear(pi_e, v, s, p, pad=0.6, n=400):
    """
    Simple grid-based equilibrium finder for the static AD–SRAS system.
    Returns (y_star, pi_star, y_grid, pi_ad, pi_sras).
    """
    if p["a1"] <= 1.0:
        print("Warning: a1 <= 1 (weak policy) - AD curve may be ill behaved")

    ybar = p["ybar"]

    # grid around ybar, using the same scaling convention as in the instructions
    y_min = (1.0 - pad) * ybar
    y_max = (1.0 + pad) * ybar
    y = np.linspace(y_min, y_max, n)

    # AD and SRAS on the grid
    pi_ad   = ad_curve(y, p, v)
    pi_sras = sras_curve(y, p, pi_e, s)

    # pick the grid point where the two curves are closest
    gap = np.abs(pi_ad - pi_sras)
    i = int(np.argmin(gap))

    y_star = float(y[i])
    # define pi_star from one of the curves (here AD)
    pi_star = float(pi_ad[i])

    return y_star, pi_star, y, pi_ad, pi_sras


# compare with the reference solver on the same grid
y1, pi1, *_ = solve_grid_linear(pi_e=par["pi_star"], v=0.01, s=0.0, p=par, pad=0.6, n=400)
y2, pi2, *_ = solve_grid(pi_e=par["pi_star"], v=0.01, s=0.0, p=par, pad=0.6, n=400)

print(f"ours:      (y={y1:.5f}, pi={pi1:.5f})")
print(f"reference: (y={y2:.5f}, pi={pi2:.5f})")

tol_y  = 2e-3
tol_pi = 1e-2   # slightly looser for inflation, as methods differ in interpolation

assert abs(y1 - y2) < tol_y,  "y* from grid solver is too far from reference"
assert abs(pi1 - pi2) < tol_pi, "pi* from grid solver is too far from reference"
