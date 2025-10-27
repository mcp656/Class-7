# b. value-of-choice

def value_of_choice_ces(x1,alpha,beta,I,p1,p2):
    x2 = (I-p1*x1)/p2
    u = utility_ces(x1,x2,alpha,beta)
    return u

# c. objective
obj = lambda x1: -value_of_choice_ces(x1,alpha,beta,I,p1,p2)