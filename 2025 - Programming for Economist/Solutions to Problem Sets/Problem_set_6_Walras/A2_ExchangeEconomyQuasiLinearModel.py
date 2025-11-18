from types import SimpleNamespace
import numpy as np
from scipy import optimize
from ExchangeEconomyModel import ExchangeEconomyModelClass

class ExchangeEconomyQuasiLineaModelClass(ExchangeEconomyModelClass):

    # uA(x1A,x2A) = ln(x1A) + alpha*x2A
    # uB(x1B,x2B) = ln(x1B) + beta*x2B

    def __init__(self):

        """ Initialize model """

        par = self.par = SimpleNamespace()
        sol = self.sol = SimpleNamespace()
        sim = self.sim = SimpleNamespace()

        # a. preferences
        par.alpha = 1.0 # weight on good 1 consumer A
        par.beta = 1.0 # weight on good 1 consumer B

        # b. endowments
        par.w1A = 0.8 # endowment of good 1 consumer A
        par.w2A = 0.3 # endowment of good 2 consumer A

        # c. solution parameters
        par.tol = 1e-8 # tolerance for convergence
        par.maxiter = 500 # maximum number of iterations
        par.nu = 0.5 # step size for tâtonnement

        # d. random number generator
        self.rng = np.random.default_rng(12345)

    ######################
    # utility and demand # 
    ######################

    def utility_A(self,x1A,x2A):
        """
        Utility function for agent A.
        """        

        par = self.par
        return np.log(x1A) + par.alpha*x2A
    
    def x2A_indifference(self,uA,x1A):

        par = self.par
        return (uA - np.log(x1A))/par.alpha
    
    def utility_B(self,x1B,x2B):
        """
        Utility function for agent B.
        """

        par = self.par
        return np.log(x1B) + par.beta*x2B

    def x2B_indifference(self,uB,x1B):

        par = self.par
        return (uB - np.log(x1B))/par.beta

    def demand_A(self,p1):
        """
        Demand for good 1 and 2 for agent A.
        """

        par = self.par 
        income = p1*par.w1A+par.w2A
        x1 = 1.0/(par.alpha*p1)
        if income > p1*x1:
            return x1, income - p1*x1
        else:
            return income/p1, 0
    
    def demand_B(self,p1):
        """
        Demand for good 1 and 2 for agent B.
        """

        par = self.par 
        income = p1*(1-par.w1A)+(1-par.w2A)
        x1 = 1.0/(par.beta*p1)
        if income > p1*x1:
            return x1, income - p1*x1
        else:
            return income/p1, 0
        
    def solve_dictator_B(self):
        """ 
        Solve the dictator problem for agent A.
        """

        par = self.par
        sol = self.sol

        # a. objective
        def obj(xA):
            return -self.utility_B(1-xA[0],1-xA[1])

        # b. constraint
        uA_w = self.utility_A(par.w1A,par.w2A)
        constraints = ({'type': 'ineq', 'fun': lambda x: self.utility_A(x[0],x[1]) - uA_w})
        bounds = ((0,1),(0,1))
        
        # c. optimize
        opt = optimize.minimize(obj,x0=sol.xA,
                                method='SLSQP',
                                bounds=bounds,constraints=constraints)
        
        sol.xA_dictatorB = opt.x
        sol.uA_dictatorB = self.utility_A(sol.xA_dictatorB[0],sol.xA_dictatorB[1])
        sol.uB_dictatorB = -opt.fun

        print(f'Dictator solution for B:')
        print(f'x1B = {sol.xA_dictatorB[0]:12.8f}')
        print(f'x2B = {sol.xA_dictatorB[1]:12.8f}')
        print(f'Utility = {sol.uB_dictatorB:12.8f}')
            