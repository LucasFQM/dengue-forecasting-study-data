
import numpy as np
import pandas as pd


class ForecastModels:

    """
    Forecasting models based on moving averages and Monte Carlo simulation.
    """

    def SMA_model(
        self,
        series,
        window,
        df,
        n_montecarlo,
        seed,
        lower_percentile,
        upper_percentile
    ) -> pd.DataFrame:

        """
        Generate forecasts using the Simple Moving Average (SMA) model.

        Parameters
        ----------
        series : array-like
            Observed time series.
        window : int
            Number of consecutive observations used in each training window.
        df : int or float
            Degrees of freedom of the Student's t-distribution.
        n_montecarlo : int
            Number of Monte Carlo simulations per forecast.
        seed : int
            Random seed for reproducibility.
        lower_percentile : float, optional
            Lower percentile used to estimate the prediction interval.
        upper_percentile : float, optional
            Upper percentile used to estimate the prediction interval.
        Returns
        -------
        pandas.DataFrame
            DataFrame containing the observed values (Actual), point
            forecasts (Forecast), and lower and upper prediction bounds
            (Lower_95 and Upper_95) for each forecast origin.
        """

        np.random.seed(seed)
        results = []

        i = 0
        while i + window < len(series):

            train_window = series[i:i + window]
            actual = series[i + window]

            mu = np.mean(train_window)
            sigma = np.std(train_window)

            mc_forecasts = mu + sigma * np.random.standard_t(
                df, size=n_montecarlo
            )
            mc_forecasts = np.maximum(mc_forecasts, 0)

            forecast = int(np.round(np.mean(mc_forecasts)))
            lower = int(np.floor(np.percentile(
                mc_forecasts, lower_percentile
            )))
            upper = int(np.ceil(np.percentile(
                mc_forecasts, upper_percentile
            )))

            results.append({
                "Actual": actual,
                "Forecast": forecast,
                "Lower_95": lower,
                "Upper_95": upper
            })

            i += 1

        return pd.DataFrame(results)

    def compute_EMA(self, train_window, alpha) -> float:

        """
        Compute the Exponential Moving Average (EMA) of a training window.

        Parameters
        ----------
        train_window : array-like
            Observations within the current training window.
        alpha : float
            Smoothing factor controlling the relative weight assigned to
            the most recent observation. Must be between 0 and 1.

        Returns
        -------
        float
            EMA value obtained for the training window.
        """

        prev = 0.5 * (train_window[-2] + train_window[-3])

        for i in range(len(train_window) - 2, len(train_window)):
            z_t = train_window[i]
            ema_t = alpha * z_t + (1 - alpha) * prev
            prev = ema_t

        return prev

    def EMA_model(
        self,
        series,
        window,
        df,
        n_montecarlo,
        alpha,
        seed,
        lower_percentile,
        upper_percentile
    ) -> pd.DataFrame:

        """
        Generate forecasts using the Exponential Moving Average (EMA) model.

        Parameters
        ----------
        series : array-like
            Observed time series.
        window : int
            Number of consecutive observations used in each training window.
        df : int or float
            Degrees of freedom of the Student's t-distribution.
        n_montecarlo : int
            Number of Monte Carlo simulations per forecast.
        alpha : float
            EMA smoothing factor controlling the relative weight assigned
            to the most recent observation. Must be between 0 and 1.
        seed : int
            Random seed for reproducibility.
        lower_percentile : float, optional
            Lower percentile used to estimate the prediction interval.
        upper_percentile : float, optional
            Upper percentile used to estimate the prediction interval.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing the observed values (Actual), point
            forecasts (Forecast), and lower and upper prediction bounds
            (Lower_95 and Upper_95) for each forecast origin.
        """

        np.random.seed(seed)
        results = []

        i = 0
        while i + window < len(series):

            train_window = series[i:i + window]
            actual = series[i + window]

            mu = self.compute_EMA(train_window, alpha)
            sigma = np.std(train_window)

            mc_forecasts = mu + sigma * np.random.standard_t(
                df, size=n_montecarlo
            )
            mc_forecasts = np.maximum(mc_forecasts, 0)

            forecast = int(np.round(np.mean(mc_forecasts)))
            lower = int(np.floor(np.percentile(
                mc_forecasts, lower_percentile
            )))
            upper = int(np.ceil(np.percentile(
                mc_forecasts, upper_percentile
            )))

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
    Evaluation metrics for assessing forecasting model performance.
    """

    def MAE(self, actual, forecast) -> float:

        """
        Calculate the Mean Absolute Error (MAE).

        Parameters
        ----------
        actual : array-like
            Observed values.
        forecast : array-like
            Corresponding predicted values.

        Returns
        -------
        float
            Mean absolute difference between observed and predicted values.
        """

        actual = np.array(actual)
        forecast = np.array(forecast)

        return np.mean(np.abs(actual - forecast))

    def Utheil(self, actual, forecast, epsilon) -> float:

        """
        Calculate Theil's U statistic relative to a naive one-step
        persistence forecast.

        Parameters
        ----------
        actual : array-like
            Observed values.
        forecast : array-like
            Corresponding predicted values.
        epsilon : float, optional
            Small positive constant added to the denominator to prevent
            division by zero.

        Returns
        -------
        float
            Ratio between the model RMSE and the naive forecast RMSE.
        """

        actual = np.array(actual)
        forecast = np.array(forecast)

        y_true = actual[1:]
        y_pred = forecast[1:]
        y_naive = actual[:-1]

        rmse_model = np.sqrt(np.mean((y_true - y_pred) ** 2))
        rmse_naive = np.sqrt(np.mean((y_true - y_naive) ** 2))

        return rmse_model / (rmse_naive + epsilon)

    def MSE(self, actual, forecast) -> float:

        """
        Calculate the Mean Squared Error (MSE).

        Parameters
        ----------
        actual : array-like
            Observed values.
        forecast : array-like
            Corresponding predicted values.

        Returns
        -------
        float
            Mean squared difference between observed and predicted values.
        """

        actual = np.array(actual)
        forecast = np.array(forecast)

        return np.mean((actual - forecast) ** 2)

    def PICP(self, actual, lower, upper) -> float:

        """
        Calculate the Prediction Interval Coverage Probability (PICP).

        Parameters
        ----------
        actual : array-like
            Observed values.
        lower : array-like
            Lower prediction interval bounds.
        upper : array-like
            Upper prediction interval bounds.

        Returns
        -------
        float
            Proportion of observations falling within their prediction
            intervals, ranging from 0 to 1.
        """

        actual = np.array(actual)
        lower = np.array(lower)
        upper = np.array(upper)

        return np.mean((actual >= lower) & (actual <= upper))

    def MPIW(self, lower, upper) -> float:

        """
        Calculate the Mean Prediction Interval Width (MPIW).

        Parameters
        ----------
        lower : array-like
            Lower prediction interval bounds.
        upper : array-like
            Upper prediction interval bounds.

        Returns
        -------
        float
            Mean difference between the upper and lower prediction bounds.
        """

        lower = np.array(lower)
        upper = np.array(upper)

        return np.mean(upper - lower)
