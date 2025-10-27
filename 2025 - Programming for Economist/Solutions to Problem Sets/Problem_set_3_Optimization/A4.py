# --- SLSQP vs Nelder–Mead on 1D budget line ---
import numpy as np, time
from scipy import optimize
from A1 import utility_ces

# parameters
# Use these baseline parameters
alpha = 0.5
beta = 0.000001
I = 10
p1 = 1
p2 = 2

eps = 1e-10

def c2_of(x1):
    return (I - p1*x1) / p2

# closed-form optimum for a precision check
kappa   = ((p1/p2) * (1 - alpha) / alpha) ** (1.0 / (beta + 1.0))   # x2/x1
x1_star = I / (p1 + p2 * kappa)

# objective along the budget line (penalize infeasible points)
def obj(x, alpha, beta):
    x1 = float(x[0])
    x2 = c2_of(x1)
    if x1 <= 0 or x1 >= I/p1 or x2 <= 0:
        return 1e12
    return -utility_ces(x1, x2, alpha, beta)

x0 = np.array([I / (2 * p1)])

# SLSQP (bounded)
t0 = time.perf_counter()
res_slsqp = optimize.minimize(
    obj, x0=x0, args=(alpha, beta),
    bounds=[(eps, I/p1 - eps)], method="SLSQP"
)
t1 = time.perf_counter()
x1_s = float(res_slsqp.x[0]); x2_s = c2_of(x1_s)
print(
    f"SLSQP       x1*={x1_s:.10f}  "
    f"u={utility_ces(x1_s, x2_s, alpha, beta):.10f}  "
    f"nit={res_slsqp.nit}  nfev={res_slsqp.nfev}  time={(t1-t0)*1e3:.1f} ms  "
    f"error={abs(x1_s - x1_star):.2e}"
)

# Nelder–Mead (unconstrained; same penalized objective)
t0 = time.perf_counter()
res_nm = optimize.minimize(obj, x0=x0, args=(alpha, beta), method="Nelder-Mead")
t1 = time.perf_counter()
x1_n = float(res_nm.x[0]); x2_n = c2_of(x1_n)
print(
    f"Nelder-Mead x1*={x1_n:.10f}  "
    f"u={utility_ces(x1_n, x2_n, alpha, beta):.10f}  "
    f"nit={res_nm.nit}  nfev={res_nm.nfev}  time={(t1-t0)*1e3:.1f} ms  "
    f"error={abs(x1_n - x1_star):.2e}"
)
