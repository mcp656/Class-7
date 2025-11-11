def k_star_with_g(alpha, s, delta, g):
    """Steady state for per-effective-worker capital with tech growth g (n=0)."""
    return (s/(delta + g))**(1/(1 - alpha))

# quick check
print("k* (g>0), alpha=1/3, s=0.2, delta=0.08, g=0.02 ->", k_star_with_g(1/3, 0.2, 0.08, 0.02))
