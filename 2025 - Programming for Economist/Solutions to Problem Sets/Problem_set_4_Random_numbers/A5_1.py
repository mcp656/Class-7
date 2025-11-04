import numpy as np

def stationary_power(P, tol=1e-12, maxit=1_000_000, mu0=None):
    """
    Power iteration for a row-stochastic transition matrix P.
    Returns (pi, iters) where pi is the stationary distribution (1D array).
    """
    P = np.asarray(P, dtype=float)
    n = P.shape[0]

    # Minimal sanity checks
    assert P.ndim == 2 and P.shape[0] == P.shape[1], "P must be square."
    assert np.allclose(P.sum(axis=1), 1.0, atol=1e-12), "Rows must sum to 1."
    assert np.all((P >= -1e-15) & (P <= 1 + 1e-15)), "Entries must be in [0,1]."

    # Start from uniform if none provided
    if mu0 is None:
        mu = np.full(n, 1.0/n)
    else:
        mu = np.asarray(mu0, dtype=float)
        mu = mu / mu.sum()

    # Iterate until the vector stops changing
    for it in range(1, maxit + 1):
        mu_next = mu @ P
        if np.max(np.abs(mu_next - mu)) < tol:
            mu = mu_next / mu_next.sum()  # normalize for safety
            return mu, it
        mu = mu_next

    # If not converged within maxit, return last iterate (normalized)
    mu = mu / mu.sum()
    return mu, maxit
