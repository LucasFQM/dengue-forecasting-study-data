import numpy as np
import pandas as pd


class ForecastModels:
    """
    Forecasting models using Monte Carlo simulation with moving averages.
    """

    def SMA_model(self, series, window, df, n_montecarlo, seed=42):
        """
        Simple Moving Average (SMA) model with Monte Carlo simulation.

        Parameters
        ----------
        series : array-like
            Time series data.
        
        window : int
            Size of the moving window for training.

        df : int
            Degrees of freedom
        
        n_montecarlo : int
            Number of Monte Carlo simulations.

        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing 'Actual', 'Forecast', 
            'Lower_95', and 'Upper_95' for each prediction.
        """

        np.random.seed(seed)
        results = []

        i = 0
        while i + window < len(series):

            train_window = series[i:i + window]
            actual = series[i + window]

            mu = np.mean(train_window)
            sigma = np.std(train_window)

            mc_forecasts = mu + sigma * np.random.standard_t(df, size=n_montecarlo)
            mc_forecasts = np.maximum(mc_forecasts, 0)

            forecast = int(np.round(np.mean(mc_forecasts)))
            lower = int(np.floor(np.percentile(mc_forecasts, 2.5)))
            upper = int(np.ceil(np.percentile(mc_forecasts, 97.5)))

            results.append({
                "Actual": actual,
                "Forecast": forecast,
                "Lower_95": lower,
                "Upper_95": upper
            })

            i += 1

        return pd.DataFrame(results)

    def compute_EMA(self, train_window, eta):
        """
        Compute Exponential Moving Average (EMA) for a training window.

        Parameters
        ----------
        train_window : array-like
            Training window data.
        
        eta : float
            Smoothing parameter (between 0 and 1).

        Returns
        -------
        float
            EMA value for the next period.
        """

        prev = 0.5 * (train_window[-2] + train_window[-3])

        for i in range(len(train_window) - 2, len(train_window)):
            z_t = train_window[i]
            ema_t = eta * z_t + (1 - eta) * prev
            prev = ema_t

        return prev

    def EMA_model(self, series, window, df, n_montecarlo, eta, seed=42):
        """
        Exponential Moving Average (EMA) model with Monte Carlo simulation.

        Parameters
        ----------
        series : array-like
            Time series data.
        
        window : int
            Size of the moving window for training.

        df : int
            Degrees of freedom
        
        n_montecarlo : int
            Number of Monte Carlo simulations.
        
        eta : float
            Smoothing parameter for EMA (between 0 and 1).

        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing 'Actual', 'Forecast', 
            'Lower_95', and 'Upper_95' for each prediction.
        """

        np.random.seed(seed)
        results = []

        i = 0
        while i + window < len(series):

            train_window = series[i:i + window]
            actual = series[i + window]

            mu = self.compute_EMA(train_window, eta)
            sigma = np.std(train_window)

            mc_forecasts = mu + sigma*np.random.standard_t(df, size=n_montecarlo)
            mc_forecasts = np.maximum(mc_forecasts, 0)

            forecast = int(np.round(np.mean(mc_forecasts)))
            lower = int(np.floor(np.percentile(mc_forecasts, 2.5)))
            upper = int(np.ceil(np.percentile(mc_forecasts, 97.5)))

            results.append({
                "Actual": actual,
                "Forecast": forecast,
                "Lower_95": lower,
                "Upper_95": upper
            })

            i += 1

        return pd.DataFrame(results)


class Metrics:
    """
    Performance metrics for forecasting model evaluation.
    """

    def sMAPE(self, actual, forecast):
        """
        Calculate symmetric Mean Absolute Percentage Error (sMAPE).

        Parameters
        ----------
        actual : array-like
            Actual observed values.
        
        forecast : array-like
            Forecasted/predicted values.

        Returns
        -------
        float
            sMAPE value as percentage.
        """

        actual = np.array(actual)
        forecast = np.array(forecast)

        epsilon = 1e-6
        denominator = (np.abs(actual) + np.abs(forecast)) / 2 + epsilon

        smape = np.mean(np.abs(actual - forecast) / denominator) * 100
        return smape

    def MAE(self, actual, forecast):
        """
        Calculate Mean Absolute Error (MAE).

        Parameters
        ----------
        actual : array-like
            Actual observed values.
        
        forecast : array-like
            Forecasted/predicted values.

        Returns
        -------
        float
            MAE value.
        """

        actual = np.array(actual)
        forecast = np.array(forecast)

        return np.mean(np.abs(actual - forecast))

    def Utheil(self, actual, forecast):
        """
        Calculate Theil's U statistic.

        Parameters
        ----------
        actual : array-like
            Actual observed values.
        
        forecast : array-like
            Forecasted/predicted values.

        Returns
        -------
        float
            Theil's U value.
        """

        actual = np.array(actual)
        forecast = np.array(forecast)

        y_true = actual[1:]
        y_pred = forecast[1:]
        y_naive = actual[:-1]

        rmse_model = np.sqrt(np.mean((y_true - y_pred) ** 2))
        rmse_naive = np.sqrt(np.mean((y_true - y_naive) ** 2))

        return rmse_model / (rmse_naive + 1e-6)

    def MSE(self, actual, forecast):
        """
        Calculate Mean Squared Error (MSE).

        Parameters
        ----------
        actual : array-like
            Actual observed values.
        
        forecast : array-like
            Forecasted/predicted values.

        Returns
        -------
        float
            MSE value.
        """

        actual = np.array(actual)
        forecast = np.array(forecast)

        return np.mean((actual - forecast) ** 2)
