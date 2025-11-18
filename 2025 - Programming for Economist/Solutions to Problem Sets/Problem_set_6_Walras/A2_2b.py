for alpha in np.linspace(0.5,3.0,20):
    model_ = ExchangeEconomyQuasiLineaModelClass()
    model_.par.alpha = alpha
    model_.solve_walras(p_guess=1.0,print_output=False)
    print(f'{alpha:0.2f} -> Equilibrium price p1* = {model_.sol.p1:7.4f}, x1A* = {model_.sol.xA[0]:7.4f}, x2A* = {model_.sol.xA[1]:7.4f}')