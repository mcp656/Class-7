# --- CES consumer problem: Newton on F(x1)=0 with analytic derivative ---

import numpy as np

# Parameters 
# Use these baseline parameters
alpha = 0.5
beta = 0.000001
I = 10
p1 = 1
p2 = 2


def c2_of(c1, I=I, p1=p1, p2=p2):
    return (I - p1*c1)/p2

# FOC as single equation:
def F(c1, alpha=alpha, beta=beta, I=I, p1=p1, p2=p2):
    c2 = c2_of(c1, I=I, p1=p1, p2=p2)
    if c1 <= 0 or c2 <= 0:
        return np.sign(c1 - (I/p1)/2.0)  # keep search interior
    return (alpha/(1.0 - alpha)) * ((c2 / c1)**(beta + 1.0)) - (p1/p2)

# Analytic derivative:
# F'(x1) = - (alpha/(1-alpha)) * (beta+1) * [ I / (p2 * x1^2 ) ] * (x2/x1)^{beta}
def Fprime(x1, alpha=alpha, beta=beta):
    c2 = c2_of(x1)                       # <- fixed (was x2_of)
    if x1 <= 0 or c2 <= 0:
        return -1.0
    s = alpha / (1.0 - alpha)
    return - s * (beta + 1.0) * (I / (p2 * x1**2)) * ((c2 / x1)**beta)

# Minimal Newton solver
def find_root(x0, f, df, tol=1e-12, max_iter=200):
    x = float(x0)
    for it in range(1, max_iter+1):
        fx, dfx = f(x), df(x)
        if not np.isfinite(fx) or not np.isfinite(dfx) or dfx == 0:
            x = 0.5*(x + 0.5*I/p1)  # gentle fallback
            continue
        x_new = x - fx/dfx
        if x_new <= 0 or x_new >= I/p1 or not np.isfinite(x_new):
            x_new = 0.5*(x + 0.5*I/p1)
        if abs(x_new - x) < tol:
            return x_new, it
        x = x_new
    return x, max_iter

# Run Newton
x0 = 0.4 * (I/p1)
x1_newt, it_newt = find_root(x0, F, Fprime, tol=1e-12)
x2_newt = c2_of(x1_newt)

print(f"Newton: x1*={x1_newt:.10f}, x2*={x2_newt:.10f}, iterations={it_newt}")
