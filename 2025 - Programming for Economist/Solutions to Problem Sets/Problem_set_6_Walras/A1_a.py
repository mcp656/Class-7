obj = lambda p: excess_demand(p,do_print=True)
res = optimize.root_scalar(obj,method='bisect',bracket=[1,8],xtol=1e-8)

p_star = res.root
q_star = demand(p_star)

print(f'Equilibrium price: p* = {p_star:.2f}')
print(f'Equilibrium quantity: q* = {q_star:.2f}')