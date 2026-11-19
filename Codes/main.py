
import pandas as pd
from pathlib import Path
from analysis import StationarityTests, TimeSeriesAnalysis
from models import ForecastModels, Metrics


# ==================== PATHS ====================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "Data"


# ==================== DATA CONFIGURATION ====================

DATA_CONFIG = {
    "dataset": "InfoDengue",
    "cities": [
        "São José dos Campos",
        "Resende",
        "Ouro Preto",
    ],
    "start_year": 2022,
    "end_year": 2025,
    "base_year": 2014,
    "weeks_per_year": 52,
}


# ==================== ANALYSIS CONFIGURATION ====================

ANALYSIS_CONFIG = {
    "stationarity": {
        "adf": {
            "regression": "c",
        },
        "kpss": {
            "regression": "c",
        },
    },

    "stl": {
        "period": 52,
        "robust": True,
    },

    "pacf": {
        "lags": 26,
        "method": "ywm",
    },
}


# ==================== EXPERIMENT CONFIGURATION ====================

EXPERIMENT_CONFIG = {
    "model": ["SMA", "EMA"],
    "window": 3,
    "degrees_of_freedom": 3,
    "n_simulations": 300,
    "random_seed": 42,
    "alpha": 0.5
}


# ==================== FORECAST CONFIGURATION ====================

FORECAST_CONFIG = {
    "lower_percentile": 2.5,
    "upper_percentile": 97.5,
}


# ==================== METRICS CONFIGURATION ====================

METRICS_CONFIG = {
    "epsilon": 1e-6,
}


# ==================== DATA LOADING ====================

def load_data():

    """
    Load datasets for all configured cities.

    Returns
    -------
    dict
        Dictionary containing one DataFrame per city.
    """

    dataset = DATA_CONFIG["dataset"]

    return {
        city: pd.read_csv(DATA_DIR / city / f"{dataset}.csv")
        for city in DATA_CONFIG["cities"]
    }


# ==================== DATA PREPROCESSING ====================

def get_period_bounds():

    """
    Calculate the epidemiological week boundaries.

    Returns
    -------
    tuple
        Starting and ending epidemiological weeks.
    """

    start_week = (
        (DATA_CONFIG["start_year"] - DATA_CONFIG["base_year"])
        * DATA_CONFIG["weeks_per_year"] + 1
    )

    end_week = (
        (DATA_CONFIG["end_year"] - DATA_CONFIG["base_year"] + 1)
        * DATA_CONFIG["weeks_per_year"]
    )

    return start_week, end_week


def extract_period(data):

    """
    Extract cases within the configured epidemiological period.

    Parameters
    ----------
    data : dict
        Dictionary containing the city datasets.

    Returns
    -------
    dict
        Dictionary containing the extracted case series per city.
    """

    start_week, end_week = get_period_bounds()

    return {
        city: data[city].loc[
            data[city]["epi_weeks"].between(start_week, end_week),
            "cases"
        ].to_numpy()
        for city in DATA_CONFIG["cities"]
    }


# ==================== STATIONARITY TESTS ====================

def run_stationarity_tests(cases):

    """
    Perform ADF and KPSS tests for all configured cities.

    Parameters
    ----------
    cases : dict
        Dictionary containing the case series per city.

    Returns
    -------
    tuple
        DataFrames containing ADF and KPSS test results.
    """

    tester = StationarityTests()

    adf_config = ANALYSIS_CONFIG["stationarity"]["adf"]
    kpss_config = ANALYSIS_CONFIG["stationarity"]["kpss"]

    adf_results = []
    kpss_results = []

    for city in DATA_CONFIG["cities"]:

        series = cases[city]

        adf_results.append(
            tester.adf(series, city, **adf_config)
        )

        kpss_results.append(
            tester.kpss(series, city, **kpss_config)
        )

    return (
        pd.concat(adf_results, ignore_index=True),
        pd.concat(kpss_results, ignore_index=True)
    )


# ==================== TIME SERIES ANALYSIS ====================

def run_time_series_analysis(cases):

    """
    Perform STL decomposition and PACF analysis.

    Parameters
    ----------
    cases : dict
        Dictionary containing the case series per city.

    Returns
    -------
    dict
        Dictionary containing STL and PACF results per city.
    """

    analysis = TimeSeriesAnalysis()

    stl_config = ANALYSIS_CONFIG["stl"]
    pacf_config = ANALYSIS_CONFIG["pacf"]

    results = {}

    for city in DATA_CONFIG["cities"]:

        series = cases[city]

        results[city] = {
            "stl": analysis.stl_decomposition(
                series,
                **stl_config
            ),

            "pacf": analysis.pacf(
                series,
                lags=pacf_config["lags"],
                method=pacf_config["method"]
            ),
        }

    return results


# ==================== FORECASTING ====================

def run_models(cases):

    """
    Generate forecasts using the configured model.

    Parameters
    ----------
    cases : dict
        Dictionary containing the case series per city.

    Returns
    -------
    dict
        Dictionary containing forecast results per city.
    """

    models = ForecastModels()
    forecasts = {}

    model_name = EXPERIMENT_CONFIG["model"]

    if model_name not in ("SMA", "EMA"):
        raise ValueError(f"Unknown forecasting model: {model_name}")

    if model_name == "EMA" and EXPERIMENT_CONFIG["alpha"] is None:
        raise ValueError("The EMA model requires an alpha value.")

    for city in DATA_CONFIG["cities"]:

        common_params = {
            "series": cases[city],
            "window": EXPERIMENT_CONFIG["window"],
            "df": EXPERIMENT_CONFIG["degrees_of_freedom"],
            "n_montecarlo": EXPERIMENT_CONFIG["n_simulations"],
            "seed": EXPERIMENT_CONFIG["random_seed"],
            "lower_percentile": FORECAST_CONFIG["lower_percentile"],
            "upper_percentile": FORECAST_CONFIG["upper_percentile"],
        }

        if model_name == "SMA":
            forecast = models.SMA_model(**common_params)

        else:
            forecast = models.EMA_model(
                **common_params,
                alpha=EXPERIMENT_CONFIG["alpha"]
            )

        forecasts[city] = forecast

    return forecasts


# ==================== MODEL EVALUATION ====================

def evaluate_models(forecasts):

    """
    Evaluate forecasting performance using point and interval metrics.

    Parameters
    ----------
    forecasts : dict
        Dictionary containing forecast results per city.

    Returns
    -------
    pandas.DataFrame
        Forecasting performance metrics for each city.
    """

    metrics = Metrics()
    results = []

    for city, forecast in forecasts.items():

        actual = forecast["Actual"].values
        predicted = forecast["Forecast"].values

        results.append({
            "city": city,
            "MAE": metrics.MAE(actual, predicted),
            "MSE": metrics.MSE(actual, predicted),
            "Theil": metrics.Utheil(
                actual,
                predicted,
                epsilon=METRICS_CONFIG["epsilon"]
            ),
            "PICP": metrics.PICP(
                actual,
                forecast["Lower_95"].values,
                forecast["Upper_95"].values
            ),
            "MPIW": metrics.MPIW(
                forecast["Lower_95"].values,
                forecast["Upper_95"].values
            ),
        })

    return pd.DataFrame(results)


# ==================== EXPERIMENT EXECUTION ====================


# ==================== EXPERIMENT EXECUTION ====================

def run_experiment(cases, verbose=False):

    """
    Execute the forecasting experiment and evaluate its results.

    Parameters
    ----------
    cases : dict
        Dictionary containing the case series per city.
    verbose : bool
        If True, print the experiment configuration and results.

    Returns
    -------
    tuple
        Dictionary of forecasts and DataFrame of evaluation metrics.
    """

    if verbose:

        print("\n" + "=" * 60)
        print("EXPERIMENT CONFIGURATION")
        print("=" * 60)

        for title, config in [
            ("Forecasting Model", EXPERIMENT_CONFIG),
            ("Forecast Configuration", FORECAST_CONFIG),
            ("Metrics Configuration", METRICS_CONFIG),
        ]:
            print(f"\n---------- {title} ----------")
            for parameter, value in config.items():
                print(f"{parameter}: {value}")

    forecasts = run_models(cases)
    results = evaluate_models(forecasts)

    if verbose:
        print(f"\n========== {EXPERIMENT_CONFIG['model']} RESULTS ==========")
        print(results.round(2).to_string(index=False))

    return forecasts, results


# ==================== MAIN EXECUTION ====================

def main():

    """
    Execute the configured time series analysis pipeline.
    """

    data = load_data()
    cases = extract_period(data)

    # Stationarity analysis
    #adf, kpss = run_stationarity_tests(cases)

    #print("\n---------- ADF ----------")
    #print(adf.to_string(index=False))

    #print("\n---------- KPSS ----------")
    #print(kpss.to_string(index=False))

    # Time series analysis
    #ts_results = run_time_series_analysis(cases)

    #print("\n---------- PACF ----------")
    #print(ts_results["São José dos Campos"]["pacf"].round(2))

    #print("\n---------- STL Decomposition ----------")

    #print("\n---------- Trend ----------")
    #print(ts_results["São José dos Campos"]["stl"].trend.round(2))

    #print("\n---------- Seasonal ----------")
    #print(ts_results["São José dos Campos"]["stl"].seasonal.round(2))

    #print("\n---------- Residual ----------")
    #print(ts_results["São José dos Campos"]["stl"].resid.round(2))


    # EMA forecasting
    EXPERIMENT_CONFIG["model"] = "EMA"
    forecastsEMA, resultsEMA = run_experiment(cases)

    # SMA forecasting
    EXPERIMENT_CONFIG["model"] = "SMA"
    forecastsSMA, resultsSMA = run_experiment(cases)

    # Diebold-Mariano test
    analysis = TimeSeriesAnalysis()

    print("\n========== DIEBOLD-MARIANO TEST RESULTS ==========")

    for city in DATA_CONFIG["cities"]:

        dm_results = analysis.diebold_mariano(
            data=cases[city],
            forecastsEMA=forecastsEMA[city]["Forecast"].values,
            forecastsSMA=forecastsSMA[city]["Forecast"].values
        )

        print(f"\n{city}")
        print("-" * 60)
        print(dm_results.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
