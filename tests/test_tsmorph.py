import numpy as np
import pandas as pd
from tsmorph import TSmorph

def test_fit():
    S = np.array([1, 2, 3, 4, 5])
    T = np.array([6, 7, 8, 9, 10])
    granularity = 3
    morph = TSmorph(S, T, granularity)
    result = morph.fit()
    assert isinstance(result, pd.DataFrame)
    assert result.shape == (5, 3)
