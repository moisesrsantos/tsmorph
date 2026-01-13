import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
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


def test_fit_dba():
    # Slightly different shapes to exercise DBA alignment
    S = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    T = np.array([1.0, 2.0, 10.0, 11.0, 12.0])
    morph = TSmorph(S, T, granularity=3)
    result = morph.fit(use_dba=True, n_iter=5)
    assert isinstance(result, pd.DataFrame)
    assert result.shape == (5, 3)
    # Basic sanity: after DBA alignment indices may move; ensure values are within the global range of S and T
    global_min = min(S.min(), T.min())
    global_max = max(S.max(), T.max())
    for col in result.columns:
        vals = result[col].values
        assert np.all(vals >= global_min - 1e-8)
        assert np.all(vals <= global_max + 1e-8)
