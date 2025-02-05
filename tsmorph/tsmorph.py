import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pycatch22 import catch22_all
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

    def fit(self) -> pd.DataFrame:
        """
        Generates semi-synthetic time series by morphing between source and target.

        Returns:
            pd.DataFrame: Dataframe with generated morphing time series
        """
        # Garantir que ambas as séries tenham o mesmo tamanho
        min_length = min(len(self.S), len(self.T))
        self.S = self.S[-min_length:]
        self.T = self.T[-min_length:]
        
        # Criar os pesos sem incluir 0 e 1
        alpha = np.linspace(0, 1, self.granularity + 2)[1:-1]
        y_morph = {}
        
        for index, i in enumerate(alpha):
            y_morph[f"S2T_{index}"] = i * self.T + (1 - i) * self.S
        
        return pd.DataFrame(y_morph)
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