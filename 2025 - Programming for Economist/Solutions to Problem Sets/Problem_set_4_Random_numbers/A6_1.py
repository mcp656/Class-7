import numpy as np

def simulate_many_workers_shocks_3state(T, N, s, f_s, ell, f_l, s0=0, seed=42):
    """
    Vectorized many-workers simulation for a 3-state model.
    Returns states (T+1,N), the uniform draws U (T,N), and u_rate (unemp. share each t).
    """
    assert 0 <= s <= 1 and 0 <= f_s <= 1 and 0 <= ell <= 1 and 0 <= f_l <= 1
    assert f_s + ell <= 1, "In U: stay prob = 1 - f_s - ell must be >= 0"

    rng = np.random.default_rng(seed)
    states = np.full((T + 1, N), s0, dtype=np.int8)   # 0=E, 1=U, 2=L
    U = rng.random((T, N))                            # one draw per worker per period

    for t in range(T):
        cur = states[t]
        nxt = cur.copy()          # start from "stay"

        inE = (cur == 0)
        inU = (cur == 1)
        inL = (cur == 2)

        # E: with prob s -> U; else stay E
        rE = U[t, inE]
        nxt[inE] = np.where(rE < s, 1, 0)

        # U: with prob f_s -> E; else if < f_s+ell -> L; else stay U
        rU = U[t, inU]
        idxU = np.where(inU)[0]
        toE = rU < f_s
        toL = (rU >= f_s) & (rU < f_s + ell)
        nxt[idxU[toE]] = 0
        nxt[idxU[toL]] = 2
        # remaining U stay in U

        # L: with prob f_l -> E; else stay L
        rL = U[t, inL]
        idxL = np.where(inL)[0]
        toE_L = rL < f_l
        nxt[idxL[toE_L]] = 0

        states[t + 1] = nxt

    # unemployment share (U or L)
    u_rate = (states != 0).mean(axis=1)
    return states, U, u_rate

T = 10_000
N = 20_000
s   = 0.02   # E->U
f_s = 0.30   # U->E
ell = 0.10   # U->L
f_l = 0.20   # L->E
s0  = 0
seed = 7

states, Udraws, u_rate = simulate_many_workers_shocks_3state(T, N, s, f_s, ell, f_l, s0, seed)

print("First 10 unemployment rates:", np.round(u_rate[:10], 3))
print("Last 10 unemployment rates :", np.round(u_rate[-10:], 3))
print(f"Time-avg unemployment      : {u_rate.mean():.3f}")
print(f"Pooled unemployment (T×N)  : {(states != 0).mean():.3f}")
