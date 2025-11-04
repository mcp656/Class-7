import numpy as np
from A4 import make_P_2state
from A5_1 import stationary_power


# ------------------------------------------------------------
# 6.2 Implementing power iteration & comparing to closed form
# ------------------------------------------------------------
# Assumes you already defined:
#   - make_P_2state(s, f)  ->  [[1-s, s],
#                                [ f, 1-f]]
#   - stationary_power(P, tol=1e-12, maxit=1_000_000, mu0=None)
# If not, run those cells first.

# 1) Choose hazards (you can change these)
s = 0.02   # separation probability  P(E->U)
f = 0.30   # job-finding probability P(U->E)

# 2) Build the 2x2 transition matrix P for states 0=E, 1=U
P = make_P_2state(s, f)

# 3) Apply power iteration to get stationary distribution pi = (pi_E, pi_U)
#    Start from uniform by default; stop when changes are < tol.
pi, iters = stationary_power(P, tol=1e-12, maxit=1_000_000)

# 4) Extract stationary unemployment (state 1 = U)
pi_U = float(pi[1])

# 5) Closed-form benchmark: pi_U = s / (s + f)
pi_U_cf = s / (s + f)

# 6) Compare and report
diff = abs(pi_U - pi_U_cf)

print("Transition matrix P:")
print(P)
print()
print(f"Stationary distribution (power iteration): pi_E={pi[0]:.12f}, pi_U={pi[1]:.12f}")
print(f"Iterations used: {iters}")
print()
print(f"Closed-form unemployment: pi_U_cf = s/(s+f) = {pi_U_cf:.12f}")
print(f"Absolute difference      : |pi_U - pi_U_cf| = {diff:.3e}")

# 7) Optional: enforce the tolerance requirement in the task
assert diff < 1e-10, "Difference exceeds the tolerance 1e-10. Check your implementation."
