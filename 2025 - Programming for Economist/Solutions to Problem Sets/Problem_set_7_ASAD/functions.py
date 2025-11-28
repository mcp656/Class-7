import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------------------------
# Global parameter dictionary (same structure as in the lecture notebook)
# --------------------------------------------------------------------
# Unified parameter dict used for BOTH closed and open economy code
par = dict(
    # Steady state
    ybar       = 1.0,    # potential output \bar y
    pi_star    = 0.02,   # domestic inflation target \pi^*
    pi_foreign = 0.02,   # foreign inflation (open economy anchor)

    # IS curve + Taylor rule (works for closed & flexible-rate open AD)
    b   = 0.6,    # IS sensitivity to real rate
    a1  = 1.5,    # policy response to inflation gap (>1 for Taylor principle)
    a2  = 0.10,   # policy response to output gap (>=0)

    # Phillips curve (SRAS)
    gamma = 0.4,  # slope of SRAS

    # Expectations persistence (closed-economy dynamics)
    phi = 0.7,    # persistence of expected inflation

    # Openness / exchange-rate channel (open economy)
    alpha_er = 0.50,  # extra AD stabilization under flexible ER
    beta1    = 0.50,  # AD sensitivity to real exchange rate under a peg

    # Shock dynamics for simulations (both versions can use these)
    delta   = 0.80,   # AR(1) persistence of demand shocks
    omega   = 0.15,   # AR(1) persistence of supply shocks
    sigma_x = 0.01,   # std dev of demand shock innovations
    sigma_c = 0.005   # std dev of supply shock innovations
)


# --------------------------------------------------------------------
# Closed-economy AS–AD primitives
# --------------------------------------------------------------------
def ad_curve(y, p, v):
    """
    AD curve:
        π = π* - ((y - ybar) - z_t) / α,
    where α and z_t are functions of (b, a1, a2, v).

    Parameters
    ----------
    y : array_like
        Output grid.
    p : dict
        Parameter dict with keys 'b', 'a1', 'a2', 'pi_star', 'ybar'.
    v : float
        Demand shock.

    Returns
    -------
    numpy.ndarray
        Inflation values on the AD curve for each y.
    """
    alpha_val = p["b"] * (p["a1"] - 1.0) / (1.0 + p["b"] * p["a2"])
    z_t = v / (1.0 + p["b"] * p["a2"])
    return p["pi_star"] - ((y - p["ybar"]) - z_t) / alpha_val


def sras_curve(y, p, pi_e, s):
    """
    SRAS curve:
        π = π_e + γ (y - ybar) + s

    Parameters
    ----------
    y : array_like
        Output grid.
    p : dict
        Parameter dict with keys 'gamma', 'ybar'.
    pi_e : float
        Expected inflation (e.g. last period's inflation).
    s : float
        Supply shock.

    Returns
    -------
    numpy.ndarray
        Inflation values on the SRAS curve for each y.
    """
    return pi_e + p["gamma"] * (y - p["ybar"]) + s


def solve_grid(pi_e=0.02, v=0.0, s=0.0, p=par, pad=0.6, n=400):
    """
    Solve for the AS–AD equilibrium on a grid by minimizing |AD - SRAS|.

    Parameters
    ----------
    pi_e : float
        Expected inflation.
    v : float
        Demand shock.
    s : float
        Supply shock.
    p : dict
        Parameter dict (default: global par).
    pad : float
        Half-width of the y-grid around ybar.
    n : int
        Number of grid points.

    Returns
    -------
    y_star : float
        Equilibrium output.
    pi_star : float
        Equilibrium inflation (average of AD and SRAS at the closest point).
    y : numpy.ndarray
        Output grid.
    pi_ad : numpy.ndarray
        Inflation on the AD curve.
    pi_sras : numpy.ndarray
        Inflation on the SRAS curve.
    """
    y = np.linspace(p["ybar"] - pad, p["ybar"] + pad, n)  # x-axis for both curves
    pi_ad   = ad_curve(y, p, v)
    pi_sras = sras_curve(y, p, pi_e, s)
    i = np.argmin(np.abs(pi_ad - pi_sras))                # index where curves are closest
    return y[i], 0.5 * (pi_ad[i] + pi_sras[i]), y, pi_ad, pi_sras


def plot_grid(pi_e=0.02, v=0.0, s=0.0, p=par, title="AS–AD (grid)"):
    """
    Convenience function: solve on a grid and plot AD, SRAS, LRAS and equilibrium.

    Parameters
    ----------
    pi_e : float
        Expected inflation.
    v : float
        Demand shock.
    s : float
        Supply shock.
    p : dict
        Parameter dict (default: global par).
    title : str
        Figure title.
    """
    y_star, pi_star, y, pi_ad, pi_sras = solve_grid(pi_e, v, s, p)
    print(f"Equilibrium: y*={y_star:.4f}, pi*={pi_star:.4f}")

    plt.figure(figsize=(6, 4))
    plt.plot(y, pi_ad,   "--", label="AD")
    plt.plot(y, pi_sras, "-",  label="SRAS")
    plt.axvline(p["ybar"], color="k", ls=":", label="LRAS")
    plt.scatter([y_star], [pi_star], c="k", label="Eq")
    plt.xlabel("y")
    plt.ylabel("pi")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def solve_grid_fine(pi_e=0.02, v=0.0, s=0.0, p=par, pad=0.6, n=800):
    """
    Return (y*, pi*, ygrid, pi_ad, pi_sras) using linear interpolation
    to solve pi_ad(y) - pi_sras(y) = 0 on the grid.
    """
    y = np.linspace(p["ybar"] - pad, p["ybar"] + pad, n)
    pi_ad   = ad_curve(y, p, v)
    pi_sras = sras_curve(y, p, pi_e, s)
    diff = pi_ad - pi_sras

    # Try to find a sign change and interpolate
    idx = np.where(diff[:-1] * diff[1:] <= 0)[0]
    if len(idx) > 0:
        i = idx[0]
        y0, y1 = y[i], y[i+1]
        d0, d1 = diff[i], diff[i+1]
        w = d0 / (d0 - d1 + 1e-16)          # fraction toward i+1 where diff=0
        y_star = y0 + w*(y1 - y0)
        # Evaluate pi using one curve (they're equal at the root)
        pi_star = pi_ad[i] + w*(pi_ad[i+1] - pi_ad[i])
    else:
        # Fallback: closest approach (rare if pad is reasonable)
        i = np.argmin(np.abs(diff))
        y_star  = y[i]
        pi_star = pi_sras[i]                 # pick one, not the average

    return float(y_star), float(pi_star), y, pi_ad, pi_sras



# ---- 1) Simple AR(1) shock generator ----------------------------------------
def ar1_series(T, rho, sigma, seed=None, shock0=0.0):
    rng = np.random.default_rng(seed)
    e   = rng.normal(0.0, sigma, size=T)
    x   = np.empty(T); x[0] = shock0
    for t in range(1, T):
        x[t] = rho*x[t-1] + e[t]
    return x


def simulate_asad(T=80, pars=None, n=400, seed=42,
                  mode="stochastic",      # "stochastic" or "impulse"
                  which="demand",         # "demand" or "supply" if impulse
                  size=0.02,              # impulse size
                  pad=0.6):               # grid half-width for solve_grid
    """
    Dynamics (grid-based each period):
      v_t = delta*v_{t-1} + x_t
      s_t = omega*s_{t-1} + c_t
      pi_e[t] = pi[t-1]
      (y[t], pi[t]) from solve_grid(pi_e, v, s, p, pad, n)
    """
    p = pars if pars is not None else par
    T = int(T)

    # --- shocks
    if mode == "stochastic":
        v = ar1_series(T, p["delta"], p["sigma_x"], seed=seed,   shock0=0.0)
        s = ar1_series(T, p["omega"], p["sigma_c"], seed=seed+1, shock0=0.0)
    elif mode == "impulse":
        v = np.zeros(T); s = np.zeros(T)
        if which == "demand":
            v[0] = size
        elif which == "supply":
            s[0] = size
        else:
            raise ValueError("which must be 'demand' or 'supply'")
        for t in range(1, T):
            v[t] = p["delta"]*v[t-1]
            s[t] = p["omega"]*s[t-1]
    else:
        raise ValueError("mode must be 'stochastic' or 'impulse'")

    # --- containers
    y    = np.empty(T)
    pi   = np.empty(T)
    pi_e = np.empty(T)

    # start at steady state
    pi_tm1 = p["pi_star"]

    for t in range(T):
        pi_e[t] = pi_tm1                              # adaptive expectations
        y[t], pi[t], *_ = solve_grid_fine(                 # 
            pi_e=pi_e[t], v=v[t], s=s[t], p=p, pad=pad, n=n
        )
        pi_tm1 = pi[t]

    return {"y": y, "pi": pi, "v": v, "s": s, "pi_e": pi_e, "par": p}


# ----  Plot for Inflation; Output and Shocks -------------
def plot_paths(res, title_suffix=""):
    p = res["par"]
    # Inflation
    plt.figure(figsize=(6.4,3.4))
    plt.plot(res["pi"], lw=2)
    plt.axhline(p["pi_star"], ls="--", c="k", lw=1)
    plt.title(f"Inflation path {title_suffix}")
    plt.xlabel("t"); plt.ylabel("pi")
    plt.tight_layout(); plt.show()

    # Output
    plt.figure(figsize=(6.4,3.4))
    plt.plot(res["y"], lw=2)
    plt.axhline(p["ybar"], ls="--", c="k", lw=1)
    plt.title(f"Output path {title_suffix}")
    plt.xlabel("t"); plt.ylabel("y")
    plt.tight_layout(); plt.show()

    # Shocks
    plt.figure(figsize=(6.4,3.4))
    plt.plot(res["v"], label="v (demand)")
    plt.plot(res["s"], label="s (supply)")
    plt.title(f"Shocks {title_suffix}")
    plt.xlabel("t"); plt.legend()
    plt.tight_layout(); plt.show()


# -------------- Open Economy functions -----------------------------------------

def ad_params_domestic(p):
    b, a1, a2 = p["b"], p["a1"], p["a2"]
    alpha_dom = b*(a1 - 1.0) / (1.0 + b*a2)
    z_scale   = 1.0 / (1.0 + b*a2)
    return alpha_dom, z_scale

# ---- AD curve for OPEN economy: FLEX vs FIXED --------------------------------
def ad_curve_open(y, p, v, regime="flex", e_r_prev=0.0):
    """
    Returns pi(y) on AD, given demand shock v and regime.
      FLEX:  pi = pi_star - ((y - ybar) - z_t) / alpha_open
      FIXED: pi = (e_{-1}^r + pi_foreign) - ((y - ybar) - z_t)/beta1
             using e_t^r = e_{-1}^r + pi_foreign - pi_t  => rearranged AD
    """
    ybar, pi_star = p["ybar"], p["pi_star"]
    if regime == "flex":
        alpha_dom, z_scale = ad_params_domestic(p)
        z_t = v * z_scale
        alpha_open = alpha_dom + p.get("alpha_er", 0.0)
        if alpha_open <= 1e-12:
            alpha_open = 1e-12
        return pi_star - ((y - ybar) - z_t) / alpha_open
    else:  # "fixed"
        beta1 = p["beta1"]
        z_t   = v  # simplest: treat v as the AD level shock directly under a peg
        return (e_r_prev + p["pi_foreign"]) - ((y - ybar) - z_t) / beta1
    

def solve_grid_open(pi_e=None, v=0.0, s=0.0, p=par, pad=0.6, n=600, regime="flex", e_r_prev=0.0):
    """
    Find (y*, pi*) with linear interpolation where AD(y)=SRAS(y).
    For FIXED regime, pass the state e_r_prev. If pi_e is None:
        - FIXED:  pi_e = p["pi_foreign"]  (matches your SRAS markdown)
        - FLEX:   pi_e = p["pi_star"]     (neutral baseline)
    """
    if pi_e is None:
        pi_e = p["pi_foreign"] if regime == "fixed" else p["pi_star"]

    y = np.linspace(p["ybar"] - pad, p["ybar"] + pad, n)
    pi_ad   = ad_curve_open(y, p, v, regime=regime, e_r_prev=e_r_prev)
    pi_sras = sras_curve(y, p, pi_e, s)
    diff = pi_ad - pi_sras

    idx = np.where(diff[:-1]*diff[1:] <= 0)[0]
    if len(idx):
        i = idx[0]
        w = diff[i] / (diff[i] - diff[i+1] + 1e-16)
        y_star  = y[i] + w*(y[i+1] - y[i])
        pi_star = pi_ad[i] + w*(pi_ad[i+1] - pi_ad[i])
    else:
        i = int(np.argmin(np.abs(diff)))
        y_star, pi_star = y[i], pi_sras[i]
    return float(y_star), float(pi_star), y, pi_ad, pi_sras


# -------------- Shock generator and simulation per regime --------------------
def ar1_series(T, rho, sigma, seed=None, shock0=0.0):
    rng = np.random.default_rng(seed)
    e = rng.normal(0.0, sigma, size=T)
    x = np.empty(T); x[0] = shock0
    for t in range(1, T):
        x[t] = rho*x[t-1] + e[t]
    return x

def simulate_open(T=60, p=par, mode="impulse", which="demand", size=0.2,
                  regime="flex", pad=0.6, n=600, seed=1):
    """
    Simulate paths under a given regime using grid-based static equilibrium each period.
    mode: "impulse" or "stochastic"
    which (if impulse): "demand" or "supply"
    """
    T = int(T)
    # Build shocks
    if mode == "stochastic":
        v = ar1_series(T, p["delta"], p["sigma_x"], seed=seed,   shock0=0.0)
        s = ar1_series(T, p["omega"], p["sigma_c"], seed=seed+1, shock0=0.0)
    else:
        v = np.zeros(T); s = np.zeros(T)
        if which == "demand":
            v[0] = size
        else:
            s[0] = size
        for t in range(1, T):
            v[t] = p["delta"]*v[t-1]
            s[t] = p["omega"]*s[t-1]

    # Containers
    y  = np.empty(T); pi = np.empty(T); pi_e = np.empty(T)
    pi_prev = p["pi_star"]
    for t in range(T):
        pi_e[t] = pi_prev
        y[t], pi[t], *_ = solve_grid_open(pi_e=pi_e[t], v=v[t], s=s[t], p=p, pad=pad, n=n, regime=regime)
        pi_prev = pi[t]

    return {"y": y, "pi": pi, "v": v, "s": s, "pi_e": pi_e, "regime": regime, "par": p}


# -------------- Static plot function -----------------------------------------
def plot_static_open(pi_e=0.02, v=0.0, s=0.0, p=par, pad=0.6):
    ybar = p["ybar"]
    # Solve once per regime (same grid width for visual comparability)
    y_fix, pi_fix, ygrid, pi_ad_fix,  pi_sras_fix  = solve_grid_open(pi_e, v, s, p, pad, regime="fixed")
    y_flex,pi_flex,_,     pi_ad_flex, pi_sras_flex = solve_grid_open(pi_e, v, s, p, pad, regime="flex")

    fig, ax = plt.subplots(figsize=(6.6,4.6))
    # Curves
    ax.plot(ygrid, pi_ad_fix,  ls="--",  color="#8c564b", label="AD (fixed)")
    ax.plot(ygrid, pi_ad_flex, ls="--",  color="#1f77b4", label="AD (flex)")
    ax.plot(ygrid, pi_sras_fix, ls="-",  color="#2ca02c", label="SRAS")
    # LRAS
    ax.axvline(ybar, color="k", ls=":", lw=1.2, label="LRAS")
    # Equilibria
    ax.scatter([y_fix],[pi_fix],   color="#8c564b", edgecolor="k", zorder=5, label="Eq (fixed)")
    ax.scatter([y_flex],[pi_flex], color="#1f77b4", edgecolor="k", zorder=5, label="Eq (flex)")
    ax.set_xlabel("y"); ax.set_ylabel("pi"); ax.set_title("Static AS–AD: fixed vs flexible")
    ax.legend(loc="best", frameon=True)
    plt.tight_layout(); plt.show()

# -------------- Overlay IRFs: fixed vs flexible ------------------------------
def plot_irfs_compare(res_fix, res_flex, title_suffix=""):
    p = res_fix["par"]
    # Inflation
    plt.figure(figsize=(6.6,3.6))
    plt.plot(res_fix["pi"],  label="fixed",  lw=2, color="#8c564b")
    plt.plot(res_flex["pi"], label="flex",   lw=2, color="#1f77b4")
    plt.axhline(p["pi_star"], ls="--", c="k", lw=1)
    plt.title(f"Inflation path {title_suffix}")
    plt.xlabel("t"); plt.ylabel("pi"); plt.legend(); plt.tight_layout(); plt.show()

    # Output
    plt.figure(figsize=(6.6,3.6))
    plt.plot(res_fix["y"],  label="fixed",  lw=2, color="#8c564b")
    plt.plot(res_flex["y"], label="flex",   lw=2, color="#1f77b4")
    plt.axhline(p["ybar"], ls="--", c="k", lw=1)
    plt.title(f"Output path {title_suffix}")
    plt.xlabel("t"); plt.ylabel("y"); plt.legend(); plt.tight_layout(); plt.show()
