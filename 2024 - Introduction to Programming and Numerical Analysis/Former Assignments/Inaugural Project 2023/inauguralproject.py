
from types import SimpleNamespace

import numpy as np
from scipy import optimize
import pandas as pd 
import matplotlib.pyplot as plt
import warnings

class HouseholdSpecializationModelClass:

    def __init__(self):
        """ Setup model """

        # Create parameter and solution objects
        par = self.par = SimpleNamespace()
        sol = self.sol = SimpleNamespace()

        # Set references
        par.rho = 2.0
        par.nu = 0.001
        par.epsilon = 1.0
        par.omega = 0.5 

        # Set mens dispreference for household production
        par.kappa = 1.0

        # Set household production
        par.alpha = 0.5
        par.sigma = 1.0

        # Set wages
        par.wM = 1.0
        par.wF = 1.0
        par.wF_vec = np.linspace(0.8,1.2,5)

        # Set targets
        par.beta0_target = 0.4
        par.beta1_target = -0.1

        # f. solution
        sol.LM_vec = np.zeros(par.wF_vec.size)
        sol.HM_vec = np.zeros(par.wF_vec.size)
        sol.LF_vec = np.zeros(par.wF_vec.size)
        sol.HF_vec = np.zeros(par.wF_vec.size)

        sol.beta0 = np.nan
        sol.beta1 = np.nan

    def calc_utility(self,LM,HM,LF,HF):
        """ calculate utility """

        # Access class's parameter and solution objects
        par = self.par
        sol = self.sol

        # Set consumption of market goods
        C = par.wM*LM + par.wF*LF

        # Set household production
        if par.sigma == 0:
            H = min(HM,HF)
        elif par.sigma == 1:
            H = HM**(1-par.alpha)*HF**par.alpha
        else:
            H = ((1-par.alpha)*HM**((par.sigma-1)/par.sigma)+par.alpha*HF**((par.sigma-1)/par.sigma))**(par.sigma/(par.sigma-1))

        # Set total consumption utility
        Q = C**par.omega*H**(1-par.omega)
        utility = np.fmax(Q,1e-8)**(1-par.rho)/(1-par.rho)

        # Set disutility of work
        epsilon_ = 1+1/par.epsilon
        TM = LM+par.kappa*HM
        TF = LF+HF
        disutility = par.nu*(TM**epsilon_/epsilon_+TF**epsilon_/epsilon_)
        
        return utility - disutility

    def solve_discrete(self,do_print=False):
        """ Solve model discretely """
        
        # Access class's parameter and solution objects
        par = self.par
        sol = self.sol

        opt = SimpleNamespace()
        
        # Set the possible choices in half hours
        x = np.linspace(0,24,49)
        LM,HM,LF,HF = np.meshgrid(x,x,x,x)
    
        # Unravel the choice sets
        LM = LM.ravel()
        HM = HM.ravel()
        LF = LF.ravel()
        HF = HF.ravel()

        # Calculate the utility
        u = self.calc_utility(LM,HM,LF,HF)
    
        # Set utility to to minus infinity if constraint is broken
        I = (LM+HM > 24) | (LF+HF > 24) # | is "or"
        u[I] = -np.inf
    
        # Find the maximizing argument
        j = np.argmax(u)
        
        # Save the values which maximizes utility
        opt.LM = LM[j]
        opt.HM = HM[j]
        opt.LF = LF[j]
        opt.HF = HF[j]

        # Print the solutions
        if do_print:
            for k,v in opt.__dict__.items():
                print(f'{k} = {v:6.4f}')
        
        # Return optimal values
        return opt 
    
    def solve(self,do_print=False):
        """ Solve model continously """

        # Access class's parameter and solution objects
        par = self.par
        sol = self.sol 
        opt = SimpleNamespace()

        # Set the objective function
        def objective(x):
            return -self.calc_utility(x[0],x[1],x[2],x[3])
        
        # Create constraints for M and F
        constraint_M = ({'type': 'ineq', 'fun': lambda x: 24 - x[0] - x[1]})
        constraint_F = ({'type': 'ineq', 'fun': lambda x: 24 - x[2] - x[3]})

        constraints = (constraint_M, constraint_F)

        # Set bounds
        bounds = ((0,24),(0,24),(0,24),(0,24))

        # Create an initial guess
        initial_guess = [12, 12, 12, 12]

        # Find the solutions using solver optimize minimize 
        sol = optimize.minimize(fun = objective, x0 = initial_guess, method = 'SLSQP', bounds=bounds,
                                 constraints=constraints, tol = 1e-10)

        # Save the values which maximizes utility
        opt.LM = sol.x[0]
        opt.HM = sol.x[1]
        opt.LF = sol.x[2]
        opt.HF = sol.x[3]

        # Print the solutions
        if do_print:
            for k,v in opt.__dict__.items():
                print(f'{k} = {v:6.4f}')
        
        return opt
    
    def solve_wF_vec(self,discrete=False):
        """ Solve model for female wage vector """

        # Access class's parameter and solution objects
        par = self.par
        sol = self.sol

        # Iterate over each wage in the female wage vector
        for i, wF in enumerate(par.wF_vec):

            # Assign current wage to the parameter object's wage
            par.wF = wF

            # Solve problem using 'solve_discrete' or 'solve' method based on 'discrete' value
            opt = self.solve_discrete() if discrete else self.solve()

            # Store the solution in the corresponding index of solution object's vectors
            sol.LM_vec[i] = opt.LM
            sol.HM_vec[i] = opt.HM
            sol.LF_vec[i] = opt.LF
            sol.HF_vec[i] = opt.HF

    def run_regression(self):
        """ Run regression """

        # Access class's parameter and solution objects
        par = self.par
        sol = self.sol

        # Create independent and dependent variables for regression
        x = np.log(par.wF_vec)
        y = np.log(sol.HF_vec/sol.HM_vec)

        # Arrange data for regression 
        A = np.vstack([np.ones(x.size),x]).T

        # Perform regression and save coefficients
        sol.beta0, sol.beta1 = np.linalg.lstsq(A,y,rcond=None)[0]

    def estimate(self):
        """ estimate alpha and sigma """

        # Access class's parameter and solution objects
        par = self.par
        sol = self.sol 

        # Set the objective function
        def objective(x):
            par.alpha = x[0]
            par.sigma = x[1]

            self.solve_wF_vec()
            
            self.run_regression()
            return (par.beta0_target - sol.beta0)**2 + (par.beta1_target - sol.beta1)**2
        
        # Set bounds
        bounds = ((0.5, 0.99), (0.01, 0.33))

        # Create an initial guess
        initial_guess = [0.8, 0.1] 

        # Find the solutions using solver optimize minimize 
        solution = optimize.minimize(fun = objective, x0 = initial_guess, method = 'Nelder-Mead', bounds = bounds)

        # Save the values which minimizes the squared residuals
        sol.alpha = solution.x[0]
        sol.sigma = solution.x[1]

        return sol
    
    def estimate_sigma_kappa(self, alpha=0.5):
        """ estimate sigma and kappa with fixed alpha """

        # Access class's parameter and solution objects
        par = self.par
        sol = self.sol 

        # Set the objective function
        def objective(x):
            par.alpha = alpha # fixed alpha
            par.sigma = x[0] # estimate sigma
            par.kappa = x[1] # estimate kappa

            self.solve_wF_vec()

            self.run_regression()
            return (par.beta0_target - sol.beta0)**2 + (par.beta1_target - sol.beta1)**2
        
        # Set bounds
        bounds = [(0.01, 0.33), (1, 40)] # bounds for sigma and kappa

        # Create an initial guess
        initial_guess = [0.1, 10.0] # initial guess for sigma and kappa

        # Find the solutions using solver optimize minimize 
        solution = optimize.minimize(fun = objective, x0 = initial_guess, method = 'Nelder-Mead', bounds = bounds)
        
        # Save the values which minimizes the squared residuals
        sol.alpha = alpha
        sol.sigma = solution.x[0] # estimated sigma
        sol.kappa = solution.x[1] # estimated kappa

        return sol