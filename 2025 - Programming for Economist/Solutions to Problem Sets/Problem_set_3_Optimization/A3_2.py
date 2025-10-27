import numpy as np


# Use these baseline parameters
alpha = 0.5
beta = 0.000001
I = 10
p1 = 1
p2 = 2

## required to redefine

def bisection(f, a, b, tol=1e-10, max_iter=500):
    fa, fb = f(a), f(b)
    if np.isnan(fa) or np.isnan(fb):
        raise ValueError("f(a) or f(b) is NaN.")
    if fa == 0.0: 
        return a, 0
    if fb == 0.0: 
        return b, 0
    if fa*fb > 0:
        raise ValueError("Bisection: root not bracketed. Choose a,b with opposite signs.")
    it = 0
    while (b - a) > tol and it < max_iter:
        m  = 0.5*(a + b)
        fm = f(m)
        if fm == 0.0:
            a = b = m
            break
        if fa*fm < 0:
            b, fb = m, fm
        else:
            a, fa = m, fm
        it += 1
    return 0.5*(a + b), it




# Set up the function F(c1) = 0 to solve for c1*
## write your functions here
# Budget-implied c2(c1)
def c2_of(c1, I=I, p1=p1, p2=p2):
    return (I - p1*c1)/p2

# FOC in one equation: MRS(c1) - p1/p2 = 0, where for CES
# MRS = (alpha/(1-alpha)) * (c1/c2)^(beta - 1)
def F(c1, alpha=alpha, beta=beta, I=I, p1=p1, p2=p2):
    c2 = c2_of(c1, I=I, p1=p1, p2=p2)
    # keep bisection inside the feasible open interval
    if c1 <= 0 or c2 <= 0:
        # return a sign that nudges the search back inside
        return np.sign(c1 - (I/p1)/2.0)
    return (alpha/(1.0 - alpha)) * ((c1 / c2)**(beta - 1.0)) - (p1/p2)

# Choose a bracket around the known solution
eps   = 1e-8
a_bis = eps
b_bis = I/p1 - eps

# set the tolerance and the maximum number of iterations
tol       = 1e-15
max_iter  = 500

# run the bisection method and print the solution
x1_bis, it_bis = bisection(F, a_bis, b_bis, tol=tol, max_iter=max_iter)
x2_bis = c2_of(x1_bis, I=I, p1=p1, p2=p2)
print(f"Bisection: x1*={x1_bis:.10f}, x2*={x2_bis:.10f}, iterations={it_bis}")