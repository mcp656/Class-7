import numpy as np
from typing import Callable, Union

ArrayLike = Union[float, np.ndarray, Callable[[int, float], float]]

def f(k: float, alpha: float) -> float:
    """Cobb–Douglas in intensive form (per effective worker)."""
    return k**alpha

def _val_at(x: ArrayLike, t: int, k_eff_t: float) -> float:
    """
    Resolve a parameter value at time t given a spec x:
      - float: constant
      - array-like: time path; if t exceeds length, use last element
      - callable: x(t, k_eff_t) -> float
    """
    if callable(x):
        return float(x(t, k_eff_t))
    arr = np.asarray(x)
    if arr.ndim == 0:
        return float(arr)
    idx = t if t < arr.shape[0] else -1
    return float(arr[idx])

def simulate_solow_gn(
    K0: float,
    A0: float,
    L0: float,
    T: int,
    alpha: float = 1/3,
    s: ArrayLike = 0.20,
    delta: ArrayLike = 0.08,
    g: ArrayLike = 0.02,
    n: ArrayLike = 0.01,
):
    """
    Simulate Solow with **technology growth** g and **population growth** n.
    Levels dynamics:
        Y_t = K_t^alpha * (A_t L_t)^(1-alpha)
        K_{t+1} = s_t * Y_t + (1 - delta_t) * K_t
        A_{t+1} = (1 + g_t) * A_t
        L_{t+1} = (1 + n_t) * L_t
    Returns a dict with levels and per-(effective worker, worker) series and realized paths.
    Flexible inputs:
      - s, delta, g, n each accept: float | array-like | callable (t, k_eff_t) -> float
    Notes:
      - k_eff_t = K_t / (A_t L_t) is passed to callables for policy rules.
      - Arrays may have length T or T+1; if shorter than needed, last value is reused.
    """
    # Allocate
    K = np.empty(T+1); A = np.empty(T+1); L = np.empty(T+1); Y = np.empty(T+1); C = np.empty(T+1)
    s_used = np.empty(T+1); d_used = np.empty(T+1); g_used = np.empty(T+1); n_used = np.empty(T+1)

    # Init
    K[0], A[0], L[0] = float(K0), float(A0), float(L0)

    # Simulate
    for t in range(T):
        k_eff_t = K[t] / (A[t] * L[t])
        s_used[t] = _val_at(s, t, k_eff_t)
        d_used[t] = _val_at(delta, t, k_eff_t)
        g_used[t] = _val_at(g, t, k_eff_t)
        n_used[t] = _val_at(n, t, k_eff_t)

        Y[t] = (K[t]**alpha) * (A[t]*L[t])**(1 - alpha)
        C[t] = (1.0 - s_used[t]) * Y[t]

        K[t+1] = s_used[t] * Y[t] + (1.0 - d_used[t]) * K[t]
        A[t+1] = (1.0 + g_used[t]) * A[t]
        L[t+1] = (1.0 + n_used[t]) * L[t]

    # Last-period accounting
    k_eff_T = K[-1] / (A[-1] * L[-1])
    s_used[-1] = _val_at(s, T, k_eff_T)
    d_used[-1] = _val_at(delta, T, k_eff_T)
    g_used[-1] = _val_at(g, T, k_eff_T)
    n_used[-1] = _val_at(n, T, k_eff_T)

    Y[-1] = (K[-1]**alpha) * (A[-1]*L[-1])**(1 - alpha)
    C[-1] = (1.0 - s_used[-1]) * Y[-1]

    # Derived series
    k_eff = K / (A * L)    # per effective worker
    y_eff = Y / (A * L)
    k_pw  = K / L          # per worker
    y_pw  = Y / L

    return {
        "K": K, "A": A, "L": L, "Y": Y, "C": C,
        "k_eff": k_eff, "y_eff": y_eff, "k_per_worker": k_pw, "y_per_worker": y_pw,
        "s_path": s_used, "delta_path": d_used, "g_path": g_used, "n_path": n_used,
    }

# ------------------------------
# Minimal smoke test / example
# ------------------------------
if __name__ == "__main__":
    out = simulate_solow_gn(
        K0=1.0, A0=1.0, L0=1.0, T=50,
        alpha=1/3, s=0.20, delta=0.08,
        g=0.02, n=0.01
    )
    # Example access:
    for key in ["K", "A", "L", "Y", "k_eff", "k_per_worker"]:
        print(key, "len:", len(out[key]))
