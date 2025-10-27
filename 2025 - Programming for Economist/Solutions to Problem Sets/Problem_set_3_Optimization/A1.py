def utility_ces(x1,x2,alpha,beta):
    if x1 > 0 and x2 > 0:
        return (alpha*x1**(-beta)+(1-alpha)*x2**(-beta))**(-1/beta) 