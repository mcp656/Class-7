import numpy as np
import matplotlib.pyplot as plt
from functions import par, simulate_asad ,plot_paths

# SOLUTION 5.3 — Impulse vs stochastic runs (same seed, different mode)

imp_d = simulate_asad(
    T=60,
    pars=par,
    mode="impulse",
    which="demand",
    size=0.02,
    seed=7,
)

sto = simulate_asad(
    T=200,
    pars=par,
    mode="stochastic",
    seed=7,
)

plot_paths(imp_d, title_suffix="(impulse: demand +0.02 at t=0)")
plot_paths(sto,   title_suffix="(stochastic)")
print("5.3: Plotted impulse vs stochastic simulations for comparison.")
