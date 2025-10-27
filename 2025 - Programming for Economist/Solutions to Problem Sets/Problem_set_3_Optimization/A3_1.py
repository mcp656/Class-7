from A1 import utility_ces # your CES utility function from A1.py
import numpy as np
from types import SimpleNamespace
from grid_solve import print_solution

# Use these baseline parameters
alpha = 0.5
beta = 0.000001
I = 10
p1 = 1
p2 = 2

def find_best_choice(u_func,alpha,beta,I,p1,p2,N1,N2,do_print=True):
    
    # a. allocate numpy arrays
    shape_tuple = (N1,N2)
    x1_values = np.empty(shape_tuple)
    x2_values = np.empty(shape_tuple)
    u_values = np.empty(shape_tuple)
    
    # b. start from guess of x1=x2=0
    x1_best = 0
    x2_best = 0
    u_best = -np.inf                      # CHANGED: robust init (avoid None)

    # c. loop through all possibilities
    for i in range(N1):
        for j in range(N2):
            
            # i. x1 and x2 (chained assignment)
            x1_values[i,j] = x1 = (i/(N1-1))*I/p1
            x2_values[i,j] = x2 = (j/(N2-1))*I/p2
            
            # ii. utility
            if p1*x1 + p2*x2 <= I:  # feasible
                u_values[i,j] = u_func(x1,x2,alpha=alpha, beta=beta)  # (keep your API)
            else:                   # infeasible
                u_values[i,j] = -np.inf           # CHANGED: no call that might return None
            
            # iii. check if best so far
            if u_values[i,j] > u_best:
                x1_best = x1_values[i,j]
                x2_best = x2_values[i,j] 
                u_best = u_values[i,j]
    
    # d. print
    if do_print:
        print_solution(x1_best,x2_best,u_best,I,p1,p2)

    return SimpleNamespace(
        x1_best=x1_best, x2_best=x2_best, u_best=u_best,
        x1_values=x1_values, x2_values=x2_values, u_values=u_values
    )
# parameters for grid search
N1 = 100
N2 = 100

# run once
sol_grid = find_best_choice(utility_ces,alpha,beta,I,p1,p2,N1,N2,do_print=True)
print_solution(sol_grid.x1_best, sol_grid.x2_best, sol_grid.u_best, I, p1, p2)