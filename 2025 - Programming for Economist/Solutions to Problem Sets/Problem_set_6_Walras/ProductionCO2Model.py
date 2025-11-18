from types import SimpleNamespace

import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

class ProductionCO2ModelClass():

    def __init__(self):
        """
        Initialize the model 
        """

        par = self.par = SimpleNamespace()
        sol = self.sol = SimpleNamespace()

        # a. firms
        par.A = 1.0 # total factor productivity
        par.gamma = 0.5 # curvature of production function

        # b. households
        par.alpha = 0.3 # weight on consumption good 1
        par.nu = 1.0 # weight on disutility of labor
        par.epsilon = 2.0 # curvature of disutility of labor

        # c. government
        par.tau = 0.0 # tax on good 2
        par.T = 0.0 # lump-sum transfer
        par.kappa = 0.1 # social cost of carbon emissions in good 2

        # d. solver settings
        par.tol_p = 1e-6 # tolerance for convergence
        par.tol_households = 1e-8 # tolerance for household problem

    #########
    # firms #
    #########

    def firms(self,p1,p2):
        """
        Solve the firm problem given prices p1 and p2.
        """

        par = self.par
        sol = self.sol

        w = 1.0 # normalization

        sol.l1 = (p1*par.A*par.gamma/w)**(1/(1-par.gamma))
        sol.y1 = par.A*sol.l1**par.gamma

        sol.l2 = (p2*par.A*par.gamma/w)**(1/(1-par.gamma))
        sol.y2 = par.A*sol.l2**par.gamma

        sol.pi1 = (1-par.gamma)/par.gamma*w*(p1*par.A*par.gamma/w)**(1/(1-par.gamma))
        sol.pi2 = (1-par.gamma)/par.gamma*w*(p2*par.A*par.gamma/w)**(1/(1-par.gamma))

        assert p1*sol.y1 - w*sol.l1 - sol.pi1 < 1e-8, 'Firm 1 profit calculation error'
        assert p2*sol.y2 - w*sol.l2 - sol.pi2 < 1e-8, 'Firm 2 profit calculation error'
    
    ##############
    # households #
    ##############

    def consumption(self,l,p1,p2):
        """
        Calculate consumption levels for households given labor supply and prices p1 and p2.
        """

        par = self.par
        sol = self.sol

        w = 1.0 # normalization

        income = w*l+par.T+sol.pi1+sol.pi2

        sol.c1 = par.alpha*income/p1
        sol.c2 = (1-par.alpha)*income/(p2+par.tau)

        assert p1*sol.c1 + (p2+par.tau)*sol.c2 - income < 1e-8, 'Household budget constraint error'

    def households(self,p1,p2):
        """
        Solve the household problem given prices p1 and p2.
        """

        par = self.par
        sol = self.sol

        # a. objective
        def value_of_choice(l):
            
            self.consumption(l[0],p1,p2)
            u_c = np.log(sol.c1**par.alpha*sol.c2**(1-par.alpha))
            u_l = par.nu*l[0]**(1+par.epsilon)/(1+par.epsilon)
            U = u_c - u_l
            return -U

        # b. find optimal labor supply
        res = optimize.minimize(value_of_choice,0.5,bounds=((1e-8,None),),
                                method='L-BFGS-B',tol=par.tol_households)
        
        sol.l = res.x[0]
        sol.U = -res.fun

        # c. finalize
        self.consumption(sol.l,p1,p2)

    ######################
    # Walras equilibrium #
    ######################

    def market_clearing(self,p):
        """
        Check market clearing conditions for given prices p1 and p2.
        """
        
        par = self.par
        sol = self.sol

        # a. prices
        p1,p2 = p
        
        # b. firms
        self.firms(p1,p2)

        # c. government        
        par.T = par.tau*sol.y2 # we implicitely use y2 = c2 due to market clearing

        # d. households
        self.households(p1,p2)

        return (sol.y1-sol.c1),(sol.y2-sol.c2),(sol.l-sol.l1-sol.l2)

    def solve_grid_search(self,do_print=False,Np=10):
        """
        Solve for approximate equilibrium prices p1 and p2 over a grid.     
        """

        par = self.par 
        sol = self.sol

        # a. prep    
        obj = np.inf
        p1_ast = None
        p2_ast = None
    
        # b. grid search
        p_vec = np.linspace(0.1,2.0,Np)

        for p1 in p_vec:
            for p2 in p_vec:

                errors = self.market_clearing((p1,p2))                
                obj_now = errors[0]**2+errors[1]**2
                
                if obj_now < obj:

                    obj = obj_now
                    p1_ast = p1
                    p2_ast = p2

        # c. save
        sol.p1_approx = p1_ast
        sol.p2_approx = p2_ast
        
        # d. print
        if do_print:
            print(f'{sol.p1_approx = :.2f}, {sol.p2_approx = :.2f}')

    def solve(self,do_print=False):
        """
        Solve for equilibrium prices.
        """

        par = self.par
        sol = self.sol

        # a. objective
        def obj(logp):
            p = np.exp(logp) # ensure positivity
            errors = self.market_clearing(p)
            return errors[0],errors[1]

        # b. guess
        guess = np.log((sol.p1_approx,sol.p2_approx))

        # c. solve      
        res = optimize.root(obj,guess,method='hybr',tol=par.tol_p)

        sol.p1 = np.exp(res.x[0])
        sol.p2 = np.exp(res.x[1])

        # check
        if not res.success:
            print(res)
            print(f'{res.message = }')
            print(f'{res.fun = }')
            assert np.all(np.abs(res.fun) < par.tol_p), 'Equilibrium not found!'
            check = False
        else:
            check = True
        
        # finalize
        self.market_clearing(np.exp(res.x))
           
        # d. print
        if do_print:

            print(f'{sol.y1 = :.2f}, {sol.y2 = :.2f}, {sol.p1 = :.2f}, {sol.p2 = :.2f}')
            errors = self.market_clearing((sol.p1,sol.p2))
            for i,error in enumerate(errors):
                print(f'Error in market clearing for good {i+1}: {error:12.8f}')
        
        return check
    
    ##############
    # Government #
    ##############

    def optimal_gov(self):
        """
        Find optimal values of tau and T for the government.
        """

        par = self.par
        sol = self.sol

        # a. objective
        def SWF(tau,do_print=False,do_print_solve=False):

            par = self.par
            sol = self.sol
            
            par.tau = tau  
            self.solve_grid_search(Np=10)          
            check = self.solve(do_print=do_print_solve)
            SWF = sol.U - par.kappa*sol.y2    

            if do_print or not check: 
                print(f'{tau = :.3f}, {sol.U = :.4f}, {SWF = :.5f}')

            return -SWF    

        # b. grid search
        print('grid search:')
        taus = np.linspace(0.0,0.3,50)
        SWFs = np.zeros_like(taus)
        Us = np.zeros_like(taus)

        for i,tau in enumerate(taus):
            SWFs[i] = -SWF(tau,do_print=i%5==0)
            Us[i] = sol.U

        # bc optimal tau
        i = np.argmax(SWFs)
        tau_min = taus[i-1]
        tau_max = taus[i+1]

        res = optimize.minimize_scalar(SWF,bounds=(tau_min,tau_max),method='bounded')
        
        if not res.success:
            print(f'{res.message = }')
            assert res.success, 'Optimal tau not found!'

        print('\nEquilibrium with optimal tau:')
        SWF(res.x,do_print=True,do_print_solve=True)

        # c. plot
        fig = plt.figure(figsize=(12,5))

        ax = fig.add_subplot(1,2,1)        
        ax.plot(taus,SWFs,label='SWF');
        ax.axvline(res.x,ls='--',color='k',label=r'Optimal $\tau$')
        ax.set_xlabel(r'$\tau$')
        ax.legend()

        ax = fig.add_subplot(1,2,2)        
        ax.plot(taus,Us,label='U');
        ax.axvline(res.x,ls='--',color='k',label=r'Optimal $\tau$')
        ax.set_xlabel(r'$\tau$')
        ax.legend()
        
        fig.tight_layout()