import numpy as np
from scipy import optimize
import random

def solow_equations(variables, s_K, s_H, n, g, delta, alpha, varphi):
    """
    Args:
        variables (list or tuple): Contains two variables to be solved for: physical capital and human capital
        s_K               (float): Savings rate in physical capital
        s_H               (float): Savings rate in human capital
        n                 (float): Population growth rate
        g                 (float): TFP growth rate
        delta             (float): Depreciation rate
        alpha             (float): Output elasticity of physical capital
        varphi            (float): Output elasticity of human capital
    
    Returns:
        Solow equations for k_{t+1}-k_{t} = 0 and h_{t+1}-h_{t} = 0 
    """
    # Variables to be solved for: physical capital and human capital
    k, h = variables
    
    # Checks for edge cases, used in multi_start
    if k <= 0 or h <= 0:
        # Return a very large residual to indicate a poor solution
        return [np.inf, np.inf]

    # Set Solow equations for k_{t+1}-k_{t} = 0 and h_{t+1}-h_{t} = 0 
    solow_k = (1 / ((1 + n) * (1 + g))) * (s_K * k**alpha * h**varphi - (n + g + delta + n * g) * k)
    solow_h = (1 / ((1 + n) * (1 + g))) * (s_H * k**alpha * h**varphi - (n + g + delta + n * g) * h)

    # Return equations
    return solow_k, solow_h


def multi_start(num_guesses=100, bounds=[1e-5, 50], fun=solow_equations, args= None, method='hybr'):
    """
    Performs multi-start optimization to find the steady state solutions for k and h.
    
    Args:
        num_guesses     (int): The number of random initial guesses, default = 100
        bounds        (tuple): The bounds for the random initial guesses, default = [1e-5, 50]
        fun        (function): The function to be optimized, default = n_ss_solow
        args          (tuple): The tuple of arguments for the function, default = None
        method       (method): The optimization method to use, default = 'hybr'
    
    Returns:
        The steady state values for k, h and y, and the residual of the function
    """
    # Initialize the smallest residual as infinity
    smallest_residual = np.inf

    # Generate a list of random numbers within the specified bounds
    random_samples = list(np.random.uniform(low=bounds[0], high=bounds[1], size=num_guesses))

    # Loop through each random initial guess
    for i in range(num_guesses):
        # Select a random pair of numbers from the list of random samples
        initial_guess = random.sample(random_samples, 2)

        # Solve the optimization problem with the current initial guess
        sol = optimize.root(fun = fun, x0 = initial_guess, args = args, method = method)
        
        # Calculate the residual norm (Euclidean norm) of the current solution
        residual_norm = np.linalg.norm(sol.fun)

        # If the residual norm is smaller than the current smallest residual, update the steady state of k and h and the smallest residual
        if residual_norm < smallest_residual:
            smallest_residual = residual_norm
            steady_state_k, steady_state_h = sol.x
    
    # Return optimal solutions 
    return steady_state_k, steady_state_h, smallest_residual


def null_clines(s_K, s_H, g, n, alpha, varphi, delta, Max = 50, N = 500):
    """
    Args:
        s_K               (float): Savings rate in physical capital
        s_H               (float): Savings rate in human capital
        n                 (float): Population growth rate
        g                 (float): TFP growth rate
        delta             (float): Depreciation rate
        alpha             (float): Output elasticity of physical capital
        varphi            (float): Output elasticity of human capital
        Max               (float): Maximum value of k
        N                   (int): Number of values of k
    
    Returns:
        Null-clines for physical capital and human capital
    """

    # Create a vector for N values of k from 0 to Max 
    k_vec = np.linspace(1e-5, Max, N)

    # Create two empty N-size arrays to store the null-clines
    h_vec_k, h_vec_h = np.empty(N), np.empty(N)

    # Set root_error to False
    root_error = False

    # Iterate through each value of k in k_vec
    for i, k in enumerate(k_vec):
        
        # Determine the null-clines 
        null_k = lambda h: solow_equations((k, h), s_K, s_H, n, g, delta, alpha, varphi)[0]
        null_h = lambda h: solow_equations((k, h), s_K, s_H, n, g, delta, alpha, varphi)[1]

        try:
            # Find roots for the null-clines
            sol_k = optimize.root_scalar(f = null_k, method = 'brentq', bracket = [1e-20, 50])
            sol_h = optimize.root_scalar(f = null_h, method = 'brentq', bracket = [1e-20, 50])

            # Save the roots
            h_vec_k[i], h_vec_h[i] = sol_k.root, sol_h.root
    
        except ValueError:
            if root_error == False:
                print('Due to f(a)f(b)>0, the method failed to find roots for some or all values of k')
                root_error = True  # Set the flag to True
            h_vec_k[i], h_vec_h[i] = np.nan, np.nan

    # Return array of solutions
    return k_vec, h_vec_k, h_vec_h

def find_intersection(x, y, z):
    """
    Args: 
        x (dict): Enumerated values of x
        y (dict): Enumerated values of y 
        z (dict): Enumerated values of z
    
    Returns: 
        Value at intersection of x, y and
    """

    # Find index of intersection where the sign of (y - z) changes
    idx = np.where(np.diff(np.sign(y - z)))

    # Return value of x, y and z at index
    return x[idx[0][0]], y[idx[0][0]], z[idx[0][0]]

def simulate_growth_paths(s_H, n, g, delta, alpha, varphi, s_K, T=300, shock_time=None, shock_increase=None, incr_savings = False):
    """
    Simulates the growth paths of technology-adjusted per capita physical capital, human capital, and output given the parameters and initial conditions.

    Args:
        s_H                 (float): Savings rate in human capital
        n                   (float): Population growth rate
        g                   (float): Technological progress rate
        delta               (float): Depreciation rate
        alpha               (float): Output elasticity of physical capital
        varphi              (float): Output elasticity of human capital
        s_K                 (float): Savings rate in physical capital
        T                     (int): Number of periods, default = 300.
        shock_time  (int, optional): The time at which s_H increases. If None, there's no increase. Default = None
        shock_increase      (float): The amount by which s_H increases at the shock time. Default = 0

    Returns:
        T periods of technology-adjusted per capita physical capital, human capital, output, wages, and rents
    """
    # Set values for s_H_0 and s_H_1:
    s_H_1 = 0.0001

    # Initialize arrays to store the variables
    L = np.zeros(T)
    A = np.zeros(T)
    K = np.zeros(T)
    H = np.zeros(T)
    Y = np.zeros(T)
    k = np.zeros(T)
    h = np.zeros(T)
    y = np.zeros(T)
    w = np.zeros(T)  # wages
    r = np.zeros(T)  # rents

    # Set initial values
    L[0] = 1
    A[0] = 1
    K[0] = 1
    H[0] = 1
    Y[0] = (K[0]**alpha) * (H[0]**varphi) * (A[0]*L[0])**(1-alpha-varphi)

    k[0] = K[0] / (A[0]*L[0])
    h[0] = H[0] / (A[0]*L[0])
    y[0] = Y[0] / (A[0]*L[0])
    w[0] = (1-alpha) * Y[0] / L[0]
    r[0] = alpha * Y[0] / K[0]

    # Create a simulation
    for t in range(1, T):
        L[t] = (1 + n) * L[t-1]
        A[t] = (1 + g) * A[t-1]
        K[t] = s_K * Y[t-1] + (1 - delta) * K[t-1]

        # Increase s_H at shock time
        if t == shock_time:
            s_H += shock_increase

        if incr_savings == True:
            s_H += s_H_1 * y[t-1]

        H[t] = s_H * Y[t-1] + (1 - delta) * H[t-1]
        Y[t] = (K[t]**alpha) * (H[t]**varphi) * (A[t]*L[t])**(1-alpha-varphi)
        k[t] = K[t] / (A[t]*L[t])
        h[t] = H[t] / (A[t]*L[t])
        y[t] = Y[t] / (A[t]*L[t])
        w[t] = varphi * Y[t] / H[t]
        r[t] = alpha * Y[t] / K[t]

    return k, h, y, w, r
