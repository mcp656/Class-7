def k_star_gn(alpha, s, delta, g, n):
    """Steady state for per-effective-worker capital with both g and n (discrete exact)."""
    phi = (1 + g)*(1 + n) - (1 - delta)   # = delta + g + n + g*n
    return (s / phi)**(1/(1 - alpha))

# quick smoke test
print("k* (g,n), alpha=1/3, s=0.2, delta=0.08, g=0.02, n=0.01 ->", k_star_gn(1/3, 0.2, 0.08, 0.02, 0.01))