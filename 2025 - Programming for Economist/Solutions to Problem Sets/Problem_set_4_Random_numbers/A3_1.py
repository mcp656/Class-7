from A2_2 import simulate_many_workers_shocks
import numpy as np


def spell_lengths_step(series, value):
    """
    Return lengths of *completed* runs equal to `value` (0 or 1) in a 1-D binary array.
    By design, this ignores a trailing run at the end if it never finishes inside the sample
    (i.e., right-censored spell).
    """
    lengths = []   # will collect the lengths of each completed run
    run = 0        # current run length (number of consecutive elements == `value`)
    
    # Iterate through the series element by element
    for x in series:
        if x == value:
            # We are still in (or just entered) a run of the target value
            run += 1
        else:
            # We just hit a different value -> any ongoing run ends here
            if run > 0:
                lengths.append(run)  # record the completed run's length
                run = 0              # reset for the next potential run
    
    # Note: If the series ends while still in a run (run > 0), we IGNORE it on purpose.
    # That last spell is "right-censored" (didn't end within the observed window).
    
    return np.array(lengths, dtype=int)


def spell_lengths(states, value):
    """
    Collect spell lengths of `value` (0 or 1) across MANY workers.

    Parameters
    ----------
    states : ndarray, shape (T+1, N)
        Binary panel with time on rows and workers on columns (0=E, 1=U).
    value : int {0,1}
        Which state's spell lengths to measure.

    Returns
    -------
    np.ndarray (1D)
        A single concatenated array of all *completed* spell lengths
        across every worker (edge-censored last spells are ignored,
        as in `spell_lengths_1d`).
    """
    T1, N = states.shape          # T1 = T+1 rows (time), N columns (workers)
    all_len = []                  # will collect lengths from all workers

    # Loop over workers (each column is one worker's 1D state path)
    for j in range(N):
        # Compute runs for worker j using the 1D helper.
        # NOTE: make sure your helper is named `spell_lengths_1d`.
        lengths_j = spell_lengths_step(states[:, j], value)
        # Extend the master list with this worker's completed run lengths
        all_len.extend(lengths_j)

    # Return as a NumPy integer array
    return np.array(all_len, dtype=int)



###############
# Now let's apply it
###################

T = 10_000
N = 20_000
s = 0.02   # separation probability (E->U)
f = 0.30   # job-finding probability (U->E)
s0 = 0
seed = 7

states, Udraws, u_rate = simulate_many_workers_shocks(T, N, s, f, s0, seed)
print(f"Pooled unemployment (T×N)  : {states.mean():.3f}")  # across all obs

# spells
U_spells = spell_lengths(states, value=1)   # runs of unemployment
E_spells = spell_lengths(states, value=0)   # runs of employment

avg_U = U_spells.mean() if U_spells.size else np.nan
avg_E = E_spells.mean() if E_spells.size else np.nan

print(f"Unemployment spells: count={U_spells.size:,}, mean={avg_U:.3f}, theory~{1/f:.3f}")
print(f"Employment   spells: count={E_spells.size:,}, mean={avg_E:.3f}, theory~{1/s:.3f}")