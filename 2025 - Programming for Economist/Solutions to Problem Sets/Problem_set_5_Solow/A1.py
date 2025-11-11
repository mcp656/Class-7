import numpy as np
# === Core functions for the Solow model (per worker) ===

def f(k, alpha):
    return np.power(k, alpha)

def fprime(k, alpha):
    return alpha * np.power(k, alpha-1)

def k_star_no_growth(alpha, s, delta):
    return np.power(s / delta, 1.0 / (1.0 - alpha))


# quick test
print("k* (no growth), alpha=1/3, s=0.2, delta=0.08 ->", k_star_no_growth(1/3, 0.2, 0.08))
