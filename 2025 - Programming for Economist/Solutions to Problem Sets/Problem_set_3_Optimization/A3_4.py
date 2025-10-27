import scipy.optimize as optimize
from A1 import utility_ces 
# Parameters 
# Use these baseline parameters
alpha = 0.5
beta = 0.000001
I = 10
p1 = 1
p2 = 2

def value_of_choice_ces(x1,alpha,beta,I,p1,p2):
    x2 = (I-p1*x1)/p2
    u = utility_ces(x1,x2,alpha,beta)
    return u

# c. objective
obj = lambda x1: -value_of_choice_ces(x1,alpha,beta,I,p1,p2)

# d. solve
solution = optimize.minimize_scalar(obj,bounds=(0,I/p1))

# e. result
u_ces = -solution.fun
x1_ces = solution.x
x2_ces = (I-x1_ces*p1)/p2
print(x1_ces,x2_ces,u_ces)