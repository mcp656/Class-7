import numpy as np

T = 10_000
N = 20_000
s = 0.02   # separation probability (E->U)
f = 0.30   # job-finding probability (U->E)
s0 = 0
seed = 7

def simulate_many_workers_shocks(T, N, s, f, s0=0, seed=42):
    rng = np.random.default_rng(seed)
    states = np.full((T + 1, N), s0, dtype=np.int8)
    U = rng.random((T, N))  # one draw per worker per period

    for t in range(T):
        in_E = (states[t] == 0)
        in_U = ~in_E

        # Start from "stay"
        states[t+1] = states[t]

        # E -> U on separation with prob s
        states[t+1, in_E] = np.where(U[t, in_E] < s, 1, 0)

        # U -> E on job finding with prob f
        states[t+1, in_U] = np.where(U[t, in_U] < f, 0, 1)

    u_rate = states.mean(axis=1)  # share unemployed each period
    return states, U, u_rate

N = 20_000
states, U, u_rate = simulate_many_workers_shocks(T, N, s, f, s0, seed)
print("First 10 unemployment rates:", np.round(u_rate[:10], 3))
print("Last 10 unemployment rates :", np.round(u_rate[-10:], 3))
print("Overall unemployment rate :", np.mean(u_rate))
