import numpy as np
from functions import par, ad_curve, sras_curve

# SOLUTION
p = par
alpha = p["b"]*(p["a1"]-1.0)/(1.0 + p["b"]*p["a2"])
slope_ad_analytical   = -1.0/alpha
slope_sras_analytical = p["gamma"]

h = 1e-5
y0 = p["ybar"]
pi_e = p["pi_star"]; v=s=0.0

# AD finite difference
pi_plus  = ad_curve(y0+h, p, v)
pi_minus = ad_curve(y0-h, p, v)
slope_ad_num = (pi_plus - pi_minus)/(2*h)

# SRAS finite difference
pi_plus  = sras_curve(y0+h, p, pi_e, s)
pi_minus = sras_curve(y0-h, p, pi_e, s)
slope_sras_num = (pi_plus - pi_minus)/(2*h)

print("AD slope: analytical =", slope_ad_analytical, " | numerical =", slope_ad_num)
print("SRAS slope: analytical =", slope_sras_analytical, " | numerical =", slope_sras_num)
assert np.isclose(slope_ad_num, slope_ad_analytical, atol=1e-3)
assert np.isclose(slope_sras_num, slope_sras_analytical, atol=1e-3)
