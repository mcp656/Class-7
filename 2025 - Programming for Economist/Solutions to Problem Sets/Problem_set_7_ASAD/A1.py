from functions import par, solve_grid, plot_grid
# SOLUTION
y_star, pi_star, y, pi_ad, pi_sras = solve_grid(
    pi_e=par["pi_star"], v=0.0, s=0.0, p=par, pad=0.6, n=400
)
print(f"Baseline equilibrium: y*={y_star:.4f}, pi*={pi_star:.4f}")

plot_grid(pi_e=0.02, v=0.0, s=0.0, p=par, title="AS–AD (grid)")