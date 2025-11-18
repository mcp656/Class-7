nu = 0.1
p = 5.0
tol = 1e-8
maxiter = 100
for i in range(maxiter):
    Z = excess_demand(p)
    p = p + nu*Z
    if abs(Z) < tol: break
    print(f'Iteration {i+1:3d}: p = {p:12.8f}, Z = {Z:12.8f}')

print(f'Tâtonnement converged in {i+1} iterations to price p = {p:.8f}')