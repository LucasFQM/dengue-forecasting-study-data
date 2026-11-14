import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss, pacf
from scipy.stats import t
from statsmodels.tsa.seasonal import STL


class StationarityTests:

    """
    Statistical tests for time series stationarity.
    """

    def adf(self, series, city, regression) -> pd.DataFrame:

        """
        Perform the Augmented Dickey-Fuller (ADF) test.

        Parameters
        ----------
        series : array-like
            Time series observations.
        city : str
            City name.
        regression : {"c", "ct", "ctt", "n"}, optional
            Regression specification.

        Returns
        -------
        pandas.DataFrame
            ADF test results, including test statistic, p-value,
            selected lags, observations, and critical values.
        """

        series = pd.Series(series).dropna()

        result = adfuller(
            series,
            regression=regression,
            autolag=None
        )

        return pd.DataFrame([{
            "city": city,
            "test": "ADF",
            "statistic": round(result[0], 2),
            "p_value": round(result[1], 2),
            "lags": result[2],
            "n": result[3],
            "1%": round(result[4]["1%"], 2),
            "5%": round(result[4]["5%"], 2),
            "10%": round(result[4]["10%"], 2)
        }])

    def kpss(self, series, city, regression) -> pd.DataFrame:

        """
        Perform the Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test.

        Parameters
        ----------
        series : array-like
            Time series observations.
        city : str
            City name.
        regression : {"c", "ct"}, optional
            Regression specification.

        Returns
        -------
        pandas.DataFrame
            KPSS test results, including test statistic, p-value,
            selected lags, and critical values.
        """

        series = pd.Series(series).dropna()

        result = kpss(
            series,
            regression=regression
        )

        return pd.DataFrame([{
            "city": city,
            "test": "KPSS",
            "statistic": round(result[0], 2),
            "p_value": round(result[1], 2),
            "lags": result[2],
            "1%": round(result[3]["1%"], 2),
            "5%": round(result[3]["5%"], 2),
            "10%": round(result[3]["10%"], 2)
        }])


class TimeSeriesAnalysis:

    """
    Time series decomposition and correlation analysis.
    """

    def stl_decomposition(self, series, period, robust):

        """
        Perform Seasonal-Trend decomposition using LOESS (STL).

        Parameters
        ----------
        series : array-like
            Time series observations.
        period : int
            Seasonal period.
        robust : bool, optional
            Whether to use robust fitting.

        Returns
        -------
        STLResults
            STL decomposition results.
        """

        stl = STL(
            series,
            period=period,
            robust=robust
        )

        return stl.fit()

    def pacf(self, series, lags, method):

        """
        Calculate the Partial Autocorrelation Function (PACF).

        Parameters
        ----------
        series : array-like
            Time series observations.
        lags : int
            Maximum lag.
        method : str, optional
            Estimation method. 

        Returns
        -------
        numpy.ndarray
            Partial autocorrelation coefficients.
        """

        return pacf(
            series,
            nlags=lags,
            method=method
        )


    def diebold_mariano(self, data, forecastsEMA, forecastsSMA) -> pd.DataFrame:

        """
        Perform pairwise Diebold-Mariano (DM) tests to compare
        SMA, EMA, and Naive forecasting methods.

        Parameters
        ----------
        data : array-like
            Actual time series observations.
        forecastsEMA : array-like
            EMA forecasts.
        forecastsSMA : array-like
            SMA forecasts.

        Returns
        -------
        pandas.DataFrame
            DM test statistics and p-values for each comparison.
        """

        data = np.asarray(data)
        forecastsEMA = np.asarray(forecastsEMA)
        forecastsSMA = np.asarray(forecastsSMA)

        n = len(forecastsEMA)

        actual = data[-n:]
        naive = data[-n-1:-1]

        forecasts = {
            "SMA": forecastsSMA,
            "EMA": forecastsEMA,
            "Naive": naive
        }

        comparisons = [
            ("SMA", "EMA"),
            ("SMA", "Naive"),
            ("EMA", "Naive")
        ]

        results = []

        for model1, model2 in comparisons:

            errors1 = actual - forecasts[model1]
            errors2 = actual - forecasts[model2]

            d = errors1**2 - errors2**2

            dm_stat = np.mean(d) / np.sqrt(np.var(d, ddof=1) / len(d))

            p_value = 2 * t.sf(abs(dm_stat), df=len(d) - 1)

            results.append({
                "Models": f"{model1} vs {model2}",
                "DM Statistic": dm_stat,
                "p-value": p_value
            })

        return pd.DataFrame(results)

