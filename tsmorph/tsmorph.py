import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .utils import plot_gradient_timeseries, nmae

class TSmorph:
    """
    A class for generating semi-synthetic time series through morphing.
    """
    def __init__(self, S: np.array, T: np.array, granularity: int) -> None:
        """
        Initializes the TSmorph instance.

        Args:
            S (np.array): Source time series
            T (np.array): Target time series
            granularity (int): The number of semi-synthetic time series in the morphing process
        """
        self.S = S
        self.T = T
        self.granularity = granularity

    def fit(self, use_dba: bool = True, n_iter: int = 10) -> pd.DataFrame:
        """
        Generates semi-synthetic time series by morphing between source and target.
        If `use_dba` is True, applies DTW Barycenter Averaging (DBA) to temporally align
        the series before linear interpolation.

        Args:
            use_dba (bool): Whether to align series using DBA before morphing.
            n_iter (int): Number of iterations for the DBA algorithm.

        Returns:
            pd.DataFrame: Dataframe with generated morphing time series
        """
        # Garantir que ambas as séries tenham o mesmo tamanho
        min_length = min(len(self.S), len(self.T))
        self.S = self.S[-min_length:].astype(float)
        self.T = self.T[-min_length:].astype(float)

        # Alinhar temporalmente com DBA se requerido
        if use_dba:
            centroid = self._dba([self.S, self.T], n_iter=n_iter)
            S_aligned = self._warp_to_centroid(self.S, centroid)
            T_aligned = self._warp_to_centroid(self.T, centroid)
        else:
            S_aligned = self.S
            T_aligned = self.T

        # Criar os pesos sem incluir 0 e 1
        alpha = np.linspace(0, 1, self.granularity + 2)[1:-1]
        y_morph = {}

        for index, i in enumerate(alpha):
            y_morph[f"S2T_{index}"] = i * T_aligned + (1 - i) * S_aligned

        return pd.DataFrame(y_morph)

    def _dtw_path(self, a: np.ndarray, b: np.ndarray):
        """Computes the DTW alignment path between sequences `a` and `b`.

        Returns a list of (i, j) index pairs mapping indices of `a` to indices of `b`.
        """
        n = len(a)
        m = len(b)
        cost = np.full((n + 1, m + 1), np.inf)
        cost[0, 0] = 0.0

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                dist = (a[i - 1] - b[j - 1]) ** 2
                cost[i, j] = dist + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])

        # Backtrack to get path
        i, j = n, m
        path = []
        while i > 0 or j > 0:
            if i > 0 and j > 0:
                path.append((i - 1, j - 1))
                prevs = [(cost[i - 1, j - 1], i - 1, j - 1), (cost[i - 1, j], i - 1, j), (cost[i, j - 1], i, j - 1)]
                _, i, j = min(prevs, key=lambda x: x[0])
            elif i > 0:
                path.append((i - 1, 0))
                i -= 1
            else:
                path.append((0, j - 1))
                j -= 1

        path.reverse()
        return path

    def _dba(self, sequences: list, n_iter: int = 10):
        """Simple DBA implementation for a list of 1D sequences of equal length.

        Returns the barycenter (centroid) sequence of the same length.
        """
        # Assumes all sequences have the same length
        L = len(sequences[0])
        centroid = np.mean(np.vstack(sequences), axis=0)

        for _ in range(n_iter):
            accum = np.zeros(L)
            counts = np.zeros(L)
            for s in sequences:
                path = self._dtw_path(centroid, s)
                for i, j in path:
                    accum[i] += s[j]
                    counts[i] += 1
            mask = counts > 0
            centroid[mask] = accum[mask] / counts[mask]
        return centroid

    def _warp_to_centroid(self, seq: np.ndarray, centroid: np.ndarray):
        """Projects `seq` onto the indices of `centroid` using the DTW path.

        Returns an aligned sequence with the same length as `centroid`.
        """
        L = len(centroid)
        warped = np.zeros(L)
        counts = np.zeros(L)
        path = self._dtw_path(centroid, seq)
        for i, j in path:
            warped[i] += seq[j]
            counts[i] += 1
        mask = counts > 0
        warped[mask] = warped[mask] / counts[mask]
        # fallback to centroid values where nothing was mapped
        warped[~mask] = centroid[~mask]
        return warped
    def plot(self, df: pd.DataFrame) -> None:
            """
            Plots the generated semi-synthetic time series.

            Args:
                df (pd.DataFrame): Dataframe returned from the fit method.
            """
            plot_gradient_timeseries(df)

    def analyze_morph_performance(self, df: pd.DataFrame, model, horizon: int, seasonality: int) -> None:
        """
        Analyzes model performance on synthetic time series using time-series features and MASE.

        Args:
            df (pd.DataFrame): Dataframe of generated synthetic series from fit method.
            model: Trained forecasting model compatible with neuralforecast.
            horizon (int): Forecast horizon for testing.
        """
        feature_values = []
        nmae_values = []
        
        try:
            from pycatch22 import catch22_all
        except ImportError:
            raise ImportError("pycatch22 is required for analyze_morph_performance. Install it or avoid calling this method.")

        for col in df.columns:
            series = df[col].values
            features = catch22_all(series, short_names=True)
            feature_values.append(features['values'])
            feature_names = features['short_names']
            
            # Prepare data in NeuralForecast format
            df_forecast = pd.DataFrame({
                'unique_id': [col] * len(series),
                'ds': np.arange(len(series)),
                'y': series
            })
            
            test = df_forecast.iloc[-horizon:]
            forecast_df = model.predict(test)
            forecast = forecast_df[forecast_df['unique_id'] == col][model.models[0].__class__.__name__].values[:horizon]
            
            nmae_values.append(nmae(y=test['y'].values, y_hat=forecast))
        
        feature_values = np.array(feature_values)
        nmae_values = np.array(nmae_values)
        
        num_features = feature_values.shape[1]
        x_values = np.arange(len(df.columns))
        
        for i in range(num_features):
            plt.figure(figsize=(8, 5))
            sc = plt.scatter(x_values, feature_values[:, i], c=nmae_values, cmap='viridis', edgecolors='k')
            plt.colorbar(sc, label='NMAE')
            plt.xlabel('Granularity Level')
            plt.ylabel(feature_names[i])
            plt.title(f'{feature_names[i]} variation with NMAE')
            plt.show()