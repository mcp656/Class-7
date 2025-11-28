# SOLUTION 8.1 — supply-dominated stochastic simulations (fixed vs flex)

import numpy as np
import matplotlib.pyplot as plt
from functions import par, simulate_open, plot_irfs_compare

# tilt the open-economy parameters toward supply shocks
par_supply = {
    **par,
    "sigma_x": 0.002,   # smaller demand innovations
    "sigma_c": 0.020,   # larger supply innovations
}

T = 200
seed = 123

# stochastic simulations under fixed and flexible exchange rate regimes
res_fix_sto = simulate_open(
    T=T,
    p=par_supply,
    mode="stochastic",
    regime="fixed",
    seed=seed,
)

res_flex_sto = simulate_open(
    T=T,
    p=par_supply,
    mode="stochastic",
    regime="flex",
    seed=seed,
)
