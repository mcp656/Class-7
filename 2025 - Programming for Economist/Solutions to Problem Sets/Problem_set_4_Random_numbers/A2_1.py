import numpy as np

# Single worker hit by random shocks (0=E, 1=U)
def simulate_worker_shocks(T, s, f, s0=0, seed=42):
    """
    States: 0=E, 1=U.
    - In E: separate to U with prob s
    - In U: find a job to E with prob f
    """
    rng = np.random.default_rng(seed)
    states = np.empty(T + 1, dtype=np.int8)
    states[0] = s0
    uniforms = rng.random(T)  # one draw per period: u_t ~ U[0,1]

    for t in range(T):
        epsilon = uniforms[t]
        if states[t] == 0:                 # E today
            states[t+1] = 1 if (epsilon < s) else 0
        else:                              # U today
            states[t+1] = 0 if (epsilon < f) else 1

    return states, uniforms


T = 1_000
s = 0.02   # separation probability
f = 0.30   # job-finding probability
s0 = 0     # start employed
seed = 7

path, draws = simulate_worker_shocks(T, s, f, s0, seed)
print("First 10 shocks (uniforms):", np.round(draws[:10], 3))
print("First 10 states (0=E,1=U):", path[:11].tolist())   
print(f"Share of time in U (single worker): {path.mean():.3f}")