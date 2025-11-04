import numpy as np

def make_P_2state(s, f):
    """
    Return the 2x2 transition matrix for states 0=E (employed), 1=U (unemployed).
      s : separation probability  P(E -> U)
      f : job-finding probability P(U -> E)
    """
    assert 0.0 <= s <= 1.0 and 0.0 <= f <= 1.0, "Probabilities must be in [0,1]."
    P = np.array([[1.0 - s, s],
                  [f,       1.0 - f]], dtype=float)
    assert np.allclose(P.sum(axis=1), 1.0, atol=1e-12), "Rows must sum to 1."
    return P