import numpy as np

# --- Step 1: 3-state transition matrix ---
def make_P_3state(s, f_s, ell, f_l):
    """
    Return the 3x3 transition matrix for states 0=E, 1=U, 2=L.
      s   : P(E->U)
      f_s : P(U->E)
      ell : P(U->L)
      f_l : P(L->E)
    """
    # sanity checks
    for p in (s, f_s, ell, f_l):
        assert 0.0 <= p <= 1.0, "Probabilities must be in [0,1]."
    assert f_s + ell <= 1.0, "In U: stay prob = 1 - f_s - ell must be >= 0."

    P = np.array([
        [1.0 - s,   s,             0.0],
        [f_s,       1.0 - f_s - ell, ell],
        [f_l,       0.0,           1.0 - f_l]
    ], dtype=float)

    # row-stochastic checks
    assert np.allclose(P.sum(axis=1), 1.0, atol=1e-12), "Rows must sum to 1."
    assert np.all((P >= -1e-15) & (P <= 1 + 1e-15)), "Entries must be in [0,1]."
    return P

# --- Step 2: stationary distribution via power iteration (from earlier) ---
def stationary_power(P, tol=1e-12, maxit=1_000_000, mu0=None):
    """
    Power iteration for a row-stochastic transition matrix P.
    Returns (pi, iters) where pi is the stationary distribution (1D array).
    """
    P = np.asarray(P, dtype=float)
    n = P.shape[0]
    assert P.ndim == 2 and P.shape[0] == P.shape[1], "P must be square."
    assert np.allclose(P.sum(axis=1), 1.0, atol=1e-12), "Rows must sum to 1."
    assert np.all((P >= -1e-15) & (P <= 1 + 1e-15)), "Entries must be in [0,1]."

    mu = np.full(n, 1.0/n) if mu0 is None else np.asarray(mu0, dtype=float)
    mu = mu / mu.sum()

    for it in range(1, maxit + 1):
        mu_next = mu @ P
        if np.max(np.abs(mu_next - mu)) < tol:
            mu = mu_next / mu_next.sum()
            return mu, it
        mu = mu_next

    return mu / mu.sum(), maxit

# --- Parameters ---
s   = 0.02     # E -> U
f_s = 0.30     # U -> E
ell = 0.10     # U -> L
f_l = 0.20     # L -> E

# Build P and compute stationary distribution
P = make_P_3state(s, f_s, ell, f_l)
pi, iters = stationary_power(P, tol=1e-14)

u_star = pi[1] + pi[2]   # Step 3: steady-state unemployment

print("Transition matrix P:\n", P)
print("Stationary distribution pi (E, U, L):", np.round(pi, 6))
print("Steady-state unemployment u* (= pi_U + pi_L):", round(float(u_star), 6))
print("Converged in iterations:", iters)