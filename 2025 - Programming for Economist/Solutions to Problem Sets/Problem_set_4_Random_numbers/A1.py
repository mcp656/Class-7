import numpy as np


# Solution
rng = np.random.default_rng(2025)
u1 = rng.random(5); z1 = rng.standard_normal(5)
rng_same = np.random.default_rng(2025)
u2 = rng_same.random(5); z2 = rng_same.standard_normal(5)
u_next = rng.random(5); z_next = rng.standard_normal(5)



# Check
print("u1:", u1)
print("u2 (match u1):", u2)
print("z1:", z1)
print("z2 (match z1):", z2)
print("u_next (different):", u_next)
print("z_next (different):", z_next)
assert np.allclose(u1, u2) and np.allclose(z1, z2)
assert not np.allclose(u1, u_next) and not np.allclose(z1, z_next)

