# SOLUTION 6.2 — Simulate with a given rule and compute quadratic loss
import numpy as np
from functions import solve_grid  # assumes solve_grid is available in functions.py

def simulate_with_rule_ps(a1, a2, v, s, p, pad=0.6, n=400):
    """
    Simulate the dynamic AS–AD system for given Taylor rule coefficients (a1, a2)
    and given shock paths v_t, s_t.

    Returns arrays y_t and pi_t.
    """
    P = {**p, "a1": float(a1), "a2": float(a2)}

    T = len(v)
    y = np.empty(T)
    pi = np.empty(T)

    pi_prev = P["pi_star"]

    for t in range(T):
        y[t], pi[t], *_ = solve_grid(
            pi_e=pi_prev,
            v=v[t],
            s=s[t],
            p=P,
            pad=pad,
            n=n,
        )
        pi_prev = pi[t]

    return y, pi


def loss_quad_ps(y, pi, p, lam=0.5):
    """
    Quadratic loss over the sample:

      L = sum_t (pi_t - pi_star)^2 + lam * (y_t - ybar)^2
    """
    term_pi = (pi - p["pi_star"])**2
    term_y  = (y  - p["ybar"])**2
    return float(np.sum(term_pi + lam * term_y))
