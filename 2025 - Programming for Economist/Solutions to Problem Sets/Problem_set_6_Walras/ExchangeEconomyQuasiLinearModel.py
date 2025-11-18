from types import SimpleNamespace
import numpy as np
from scipy import optimize
from ExchangeEconomyModel import ExchangeEconomyModelClass

class ExchangeEconomyModelQuasiLinearClass(ExchangeEconomyModelClass):

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
        return np.log(x1A) + par.alpha * x2A
    
    def x2A_indifference(self,uA,x1A):
        
        par = self.par
        return (uA - np.log(x1A)) / par.alpha
    
    def utility_B(self,x1B,x2B):
        """
        Utility function for agent B.
        """
        
        par = self.par
        return np.log(x1B) + par.beta * x2B
    
    def x2B_indifference(self,uB,x1B):
        
        par = self.par
        return (uB - np.log(x1B)) / par.beta
    
    def demand_A(self,p1):
        """
        Demand for good 1 and 2 for agent A.
        """
        
        par = self.par

        income = p1 * par.w1A + par.w2A
        if income > 1/par.alpha:
            return 1/(par.alpha * p1), income - (1/par.alpha)
        else:
            return income / p1, 0.0
    
    def demand_B(self,p1):
        """
        Demand for good 1 and 2 for agent B.
        """
        
        par = self.par
        
        income = p1 * (1 - par.w1A) + (1 - par.w2A)
        if income > 1/par.beta:
            return 1/(par.beta * p1), income - (1/par.beta)
        else:
            return income / p1, 0.0
    
    def solve_dictator_A(self):

        par = self.par
        sol = self.sol

        def obj(xA):
            return -self.utility_A(xA[0], xA[1])

        uB_w = self.utility_B(1 - par.w1A, 1 - par.w2A)
        constraints = [{'type': 'ineq',
                        'fun': lambda x: self.utility_B(1-x[0],1-x[1]) - uB_w}]
        bounds = ((0,1), (0,1))

        x0 = getattr(sol, 'xA', np.array([par.w1A, par.w2A]))

        opt = optimize.minimize(obj, x0=x0,
                                method='SLSQP',
                                bounds=bounds,
                                constraints=constraints)

        sol.xA_dictatorA = opt.x
        sol.uA_dictatorA = -opt.fun
        sol.uB_dictatorA = self.utility_B(1-opt.x[0], 1-opt.x[1])

        print("Dictator solution for A:")
        print(f"x1A = {opt.x[0]:12.8f}")
        print(f"x2A = {opt.x[1]:12.8f}")
        print(f"Utility A = {sol.uA_dictatorA:12.8f}")
        print(f"Utility B = {sol.uB_dictatorA:12.8f}")


    def solve_dictator_B(self):

        par = self.par
        sol = self.sol

        def obj(xA):
            return -self.utility_B(1-xA[0],1-xA[1])

        uA_w = self.utility_A(par.w1A, par.w2A)
        constraints = [{'type': 'ineq',
                        'fun': lambda x: self.utility_A(x[0], x[1]) - uA_w}]
        bounds = ((0,1), (0,1))

        x0 = getattr(sol, 'xA', np.array([par.w1A, par.w2A]))

        opt = optimize.minimize(obj, x0=x0,
                                method='SLSQP',
                                bounds=bounds,
                                constraints=constraints)

        sol.xA_dictatorB = opt.x
        sol.uB_dictatorB = -opt.fun
        sol.uA_dictatorB = self.utility_A(opt.x[0], opt.x[1])

        x1A, x2A = opt.x
        x1B, x2B = 1-x1A, 1-x2A

        print("Dictator solution for B:")
        print(f"x1A = {x1A:12.8f}, x2A = {x2A:12.8f}")
        print(f"x1B = {x1B:12.8f}, x2B = {x2B:12.8f}")
        print(f"Utility A = {sol.uA_dictatorB:12.8f}")
        print(f"Utility B = {sol.uB_dictatorB:12.8f}")