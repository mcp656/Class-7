from types import SimpleNamespace

import numpy as np
from scipy import optimize

import matplotlib.pyplot as plt

class ExchangeEconomyModelClass:

    def __init__(self):
        """ Initialize model """

        par = self.par = SimpleNamespace()
        sol = self.sol = SimpleNamespace()
        sim = self.sim = SimpleNamespace()

        # a. preferences
        par.alpha = 1/3 # weight on good 1 consumer A
        par.beta = 2/3 # weight on good 1 consumer B

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
        return x1A**par.alpha*x2A**(1-par.alpha)
    
    def x2A_indifference(self,uA,x1A):

        par = self.par
        return (uA/x1A**par.alpha)**(1/(1-par.alpha))

    def utility_B(self,x1B,x2B):
        """
        Utility function for agent B.
        """

        par = self.par
        return x1B**par.beta*x2B**(1-par.beta)

    def x2B_indifference(self,uB,x1B):

        par = self.par
        return (uB/x1B**par.beta)**(1/(1-par.beta))

    def demand_A(self,p1):
        """
        Demand for good 1 and 2 for agent A.
        """

        par = self.par 
        income = p1*par.w1A+par.w2A
        return par.alpha * ((income)/p1), (1-par.alpha) * (income)

    def demand_B(self,p1):
        """
        Demand for good 1 and 2 for agent B.
        """

        par = self.par 
        income = p1*(1-par.w1A)+(1-par.w2A)
        return par.beta * ((income)/p1), (1-par.beta) * (income)

    ########
    # plot #
    ########

    def create_edgeworthbox(self,figsize=(6,6)):
        """
        Create edgeworth box.
        """

        par = self.par

        # a. total endowment
        w1bar = 1.0
        w2bar = 1.0

        # b. figure set up
        fig = plt.figure(figsize=figsize, dpi=100)
        ax_A = fig.add_subplot(1,1,1)

        ax_A.set_xlabel('$x_1^A$')
        ax_A.set_ylabel('$x_2^A$')

        temp = ax_A.twinx()
        temp.set_ylabel('$x_2^B$')

        ax_B = temp.twiny()
        ax_B.set_xlabel('$x_1^B$')
        ax_B.invert_xaxis()
        ax_B.invert_yaxis()

        # c. limits
        ax_A.plot([0,w1bar],[0,0],lw=2,color='black')
        ax_A.plot([0,w1bar],[w2bar,w2bar],lw=2,color='black')
        ax_A.plot([0,0],[0,w2bar],lw=2,color='black')
        ax_A.plot([w1bar,w1bar],[0,w2bar],lw=2,color='black')

        ax_A.set_xlim([-0.1, w1bar + 0.1])
        ax_A.set_ylim([-0.1, w2bar + 0.1])    
        ax_B.set_xlim([w1bar + 0.1, -0.1])
        ax_B.set_ylim([w2bar + 0.1, -0.1])

        # d. make sure ax_A is on top
        ax_A.set_zorder(ax_B.get_zorder()+1)
        ax_A.patch.set_visible(False)

        return fig, ax_A, ax_B

    def indifference_curve_A(self,ax_A,x1A,x2A,**kwargs):
        """
        Plot indifference curve for agent A through x1A,x2A.
        """

        par = self.par
        
        uA = self.utility_A(x1A,x2A)
        
        x1A_grid = np.linspace(0.001,0.999,1000)
        x2A_grid = self.x2A_indifference(uA,x1A_grid)
        
        I = (x2A_grid > 0) & (x2A_grid < 1) # only inside box
        ax_A.plot(x1A_grid[I],x2A_grid[I],**kwargs)

    def indifference_curve_B(self,ax_B,x1B,x2B,**kwargs):
        """
        Plot indifference curve for agent B through x1B,x2B.
        """

        par = self.par
        
        uB = self.utility_B(x1B,x2B)
        
        x1B_grid = np.linspace(0.001,0.999,1000)
        x2B_grid = self.x2B_indifference(uB,x1B_grid)
        
        I = (x2B_grid > 0) & (x2B_grid < 1) # only inside box
        ax_B.plot(x1B_grid[I],x2B_grid[I],**kwargs)

    def plot_improvement_set(self,ax_A):
        """
        Plots the joint improvement set.
        """

        par = self.par
        sol = self.sol
        
        uA = self.utility_A(par.w1A,par.w2A)
        uB = self.utility_B(1-par.w1A,1-par.w2A)
        
        x1A_vec = np.linspace(0.01,0.99,1000)
        x2A_vec = self.x2A_indifference(uA,x1A_vec)
        
        x1B_vec = 1 - x1A_vec
        x2B_vec = self.x2B_indifference(uB,x1B_vec)
        
        I = x2A_vec < 1-x2B_vec
        I &= x2A_vec > 0
        I &= x2A_vec < 1.0
        ax_A.fill_between(x1A_vec[I],x2A_vec[I],np.fmin(1-x2B_vec[I],1.0),color='gray',alpha=0.5)

    def plot_budget_line(self,ax_A):
        """
        Plot budget line.
        """

        par = self.par
        sol = self.sol

        x1A_grid = np.linspace(0,1,100)
        x2_A = par.w2A-sol.p1*(x1A_grid-par.w1A)
        
        I = (x2_A > 0) & (x2_A < 1)
        ax_A.plot(x1A_grid[I],x2_A[I],color='black',ls='--',label='budget line')

    def add_legend(self,ax_A,ax_B,bbox_to_anchor=(0.10,0.10)):
        """
        Add legend to the edgeworth box.
        """

        handles_A, labels_A = ax_A.get_legend_handles_labels()
        handles_B, labels_B = ax_B.get_legend_handles_labels()
        ax_A.legend(handles_A+handles_B,labels_A+labels_B,
                    bbox_to_anchor=bbox_to_anchor,loc='lower left',
                    facecolor='white',framealpha=1.0)

    ######################
    # Walras equilibrium #
    ######################
    
    def check_market_clearing(self,p1):
        """
        Checks if the market clears for good 1 and 2.
        """

        par = self.par

        x1A,x2A = self.demand_A(p1)
        x1B,x2B = self.demand_B(p1)

        eps1 = (x1A-par.w1A) + x1B-(1-par.w1A)
        eps2 = (x2A-par.w2A) + x2B-(1-par.w2A)

        return eps1,eps2
  
    def solve_walras(self,p_guess,print_output=True):
        """
        Solves for the market equilibrium price using tâtonnement (auction process).
        Starting from an initial guess p_guess, the function iterates on the price until the excess demand is close to zero.
        Each updated guess is updated by direction and size of the excess demand.
        """

        par = self.par
        sol = self.sol

        # a. initial guess
        p1 = p_guess
        
        # b. iteratre
        t = 0
        while True:

            # i. excess demand
            eps = self.check_market_clearing(p1)

            # ii. stop?
            if t >= par.maxiter: raise ValueError('Max iterations exceeded')

            if np.abs(eps[0]) < par.tol:
                
                sol.p1 = p1
                sol.xA = self.demand_A(p1)
                sol.uA = self.utility_A(sol.xA[0],sol.xA[1])
                sol.uB = self.utility_B(1-sol.xA[0],1-sol.xA[1])

                if print_output:

                    print('\nSolved!')
                    print(f' {t:3d}: p1 = {p1:12.8f}, x1A = {sol.xA[0]:12.8f}, x2A = {sol.xA[1]:12.8f}')
                    print(f' Excess demand of good 1: {eps[0]:14.8f}')
                    print(f' Excess demand of good 2: {eps[1]:14.8f}')

                break

            # iii. print
            if print_output:

                if t < 5 or t %5 == 0:
                    print(f'{t:3d}: p1 = {p1:12.8f} -> excess demand of good 1 -> {eps[0]:14.8f}',end='')
                    print(f', x1A = {self.demand_A(p1)[0]:12.8f}, x2A = {self.demand_A(p1)[1]:12.8f}',end='')
                    print(f', x1B = {self.demand_B(p1)[0]:12.8f}, x2B = {self.demand_B(p1)[1]:12.8f}')
                elif t == 5:
                    print('   ...')

            # iv. p1
            p1 = p1 + par.nu*eps[0]

            # v. increment
            t += 1

    ###########################
    # other solution concepts #
    ###########################

    def solve_social_planner(self):
        """
        Solve the social planner problem.
        """

        par = self.par
        sol = self.sol

        # a. objective
        def obj(x):
            SWF = self.utility_A(x[0],x[1])+self.utility_B(1-x[0],1-x[1])
            return -SWF
        
        # b. optimize
        bounds = ((0,1),(0,1))
        opt = optimize.minimize(obj,x0=[0.5,0.5],
                                method='SLSQP',
                                bounds=bounds)
        
        # c. solution and utility
        sol.xA_planner = opt.x
        sol.uA_planner = self.utility_A(sol.xA_planner[0],sol.xA_planner[1])
        sol.uB_planner = self.utility_B(1-sol.xA_planner[0],1-sol.xA_planner[1])

        # d. print
        print(f'Planner solution:')
        print(f'x1A = {sol.xA_planner[0]:12.8f}')
        print(f'x2A = {sol.xA_planner[1]:12.8f}')
        print(f'Utility of A: {sol.uA_planner:12.8f}')
        print(f'Utility of B: {sol.uB_planner:12.8f}')

    def solve_dictator_A(self):
        """ 
        Solve the dictator problem for agent A.
        """

        par = self.par
        sol = self.sol

        # a. objective
        def obj(xA):
            return -self.utility_A(xA[0],xA[1])

        # b. constraint
        uB_w = self.utility_B(1-par.w1A,1-par.w2A)
        constraints = ({'type': 'ineq', 'fun': lambda x: self.utility_B(1-x[0],1-x[1]) - uB_w})
        bounds = ((0,1),(0,1))
        
        # c. optimize
        opt = optimize.minimize(obj,x0=sol.xA,
                                method='SLSQP',
                                bounds=bounds,constraints=constraints)
        
        sol.xA_dictatorA = opt.x
        sol.uA_dictatorA = -opt.fun
        sol.uB_dictatorA = self.utility_B(1-sol.xA_dictatorA[0],1-sol.xA_dictatorA[1])

        print(f'Dictator solution for A:')
        print(f'x1A = {sol.xA_dictatorA[0]:12.8f}')
        print(f'x2A = {sol.xA_dictatorA[1]:12.8f}')
        print(f'Utility = {sol.uA_dictatorA:12.8f}')
    
    #####################
    # random endowments #
    #####################
    
    def draw_random_endowments(self,N=50):
        """
        Draw random endowments for the simulation.
        """

        sim = self.sim

        sim.WA = self.rng.uniform(0,1,size=(2,N))

    def solve_random_endowments(self):
        """
        Solve for each draw of endowments.
        """

        sim = self.sim
        sol = self.sol
    
        sim.xA = np.empty(sim.WA.shape)

        for i in range(sim.WA.shape[1]):

            # a. set endowments
            self.par.w1A = sim.WA[0,i]
            self.par.w2A = sim.WA[1,i]

            # b. solve 
            self.solve_walras(p_guess=sol.p1,print_output=False)
            
            # c. save
            sim.xA[:,i] = sol.xA