from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.stattools import acf, pacf, ccf
from statsmodels.tsa.seasonal import STL
import pandas as pd
import numpy as np


class StationarityTests:
    """
    Perform stationarity tests for time series data.
    """

    def adf(self, series, city) -> pd.DataFrame:
        """
        Perform Augmented Dickey-Fuller test.

        Parameters
        ----------
        series : pandas.Series or array-like
            Time series data to test.

        city : str
            Name of the city/location.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing test statistics, p-value, lags used,
            number of observations, and critical values at 1%, 5%, and 10%.
        """

        series = pd.Series(series).dropna()

        result = adfuller(series, autolag='AIC')

        return pd.DataFrame([{
            'city': city,
            'test': 'ADF',
            'statistic': result[0],
            'p_value': result[1],
            'lags': result[2],
            'n': result[3],
            'critical_1%': result[4]['1%'],
            'critical_5%': result[4]['5%'],
            'critical_10%': result[4]['10%']
        }])
    
    def kpss(self, series, city) -> pd.DataFrame:
        """
        Perform Kwiatkowski-Phillips-Schmidt-Shin test.

        Parameters
        ----------
        series : pandas.Series or array-like
            Time series data to test.

        city : str
            Name of the city/location.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing test statistics, p-value, lags used,
            and critical values at 1%, 5%, and 10%.
        """

        series = pd.Series(series).dropna()

        result = kpss(series, regression="c", nlags='auto')

        return pd.DataFrame([{
            'city': city,
            'test': 'KPSS',
            'statistic': result[0],
            'p_value': result[1],
            'lags': result[2],
            'critical_1%': result[3]['1%'],
            'critical_5%': result[3]['5%'],
            'critical_10%': result[3]['10%']
        }])


class TimeSeriesAnalysis:
    """
    Perform time series decomposition and correlation analysis.
    """

    def stl_decomposition(self, series, period):
        """
        Perform STL (Seasonal-Trend decomposition using LOESS) decomposition.

        Parameters
        ----------
        series : pandas.Series or array-like
            Time series data to decompose.

        period : int
            Seasonal period of the time series.

        Returns
        -------
        DecomposeResult
            Object containing trend, seasonal, and residual components.
        """

        stl = STL(series, period=period, robust=True)
        return stl.fit()

    def acf(self, series, lags):
        """
        Calculate Autocorrelation Function (ACF).

        Parameters
        ----------
        series : pandas.Series or array-like
            Time series data.

        lags : int or array-like
            Number of lags to compute.

        Returns
        -------
        ndarray
            Array of autocorrelation values for each lag.
        """

        return acf(series, lags, fft=False)

    def pacf(self, series, lags):
        """
        Calculate Partial Autocorrelation Function (PACF).

        Parameters
        ----------
        series : pandas.Series or array-like
            Time series data.

        lags : int or array-like
            Number of lags to compute.

        Returns
        -------
        ndarray
            Array of partial autocorrelation values for each lag.
        """

        return pacf(series, lags, method='ywm')

    def ccf(self, series, variable, lags):
        """
        Calculate Cross-Correlation Function (CCF) between two time series.

        Parameters
        ----------
        series : pandas.Series or array-like
            Time series data (cases).

        variable : pandas.Series or array-like
            Secondary time series variable (e.g., climate).

        lags : int
            Number of lags to compute.

        Returns
        -------
        ndarray
            Array of cross-correlation values for lags 0 to lags.
        """

        series = np.asarray(series).flatten()
        variable = np.asarray(variable).flatten()

        series_centered = series - np.mean(series)
        variable_centered = variable - np.mean(variable)

        return ccf(variable_centered, series_centered)[:lags + 1]