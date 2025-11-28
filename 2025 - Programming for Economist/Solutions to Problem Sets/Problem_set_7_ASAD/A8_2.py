# SOLUTION 6.2 — volatility of y and pi under fixed vs flex
import numpy as np
import matplotlib.pyplot as plt
from A8_1 import par_supply, res_fix_sto, res_flex_sto

pi_star = par_supply["pi_star"]
ybar    = par_supply["ybar"]

# optional burn-in to focus on the stationary part
burn = 20
sl = slice(burn, None)

y_fix   = res_fix_sto["y"][sl]
y_flex  = res_flex_sto["y"][sl]
pi_fix  = res_fix_sto["pi"][sl]
pi_flex = res_flex_sto["pi"][sl]

std_y_fix   = np.std(y_fix  - ybar)
std_y_flex  = np.std(y_flex - ybar)
std_pi_fix  = np.std(pi_fix  - pi_star)
std_pi_flex = np.std(pi_flex - pi_star)

print("Volatility measures (after burn-in):")
print(f"  Std dev output gap:    fixed = {std_y_fix:.4f}, flex = {std_y_flex:.4f}")
print(f"  Std dev inflation gap: fixed = {std_pi_fix:.4f}, flex = {std_pi_flex:.4f}")
