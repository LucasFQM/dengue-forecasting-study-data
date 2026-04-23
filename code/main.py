import pandas as pd
from pathlib import Path
from analysis import StationarityTests, TimeSeriesAnalysis
from models import ForecastModels, Metrics
from processing import DataProcessor

# ==================== CONFIG ====================

DATA_DIR = Path("/home/lucasmoreira/Downloads/Dengue/data")

CITIES = ["Sao_Jose_dos_Campos", "Resende", "Ouro_Preto"]
DATASETS = ["R1", "R21", "R22", "R23"]

START_YEAR = 2022
BASE_YEAR = 2014
WEEKS_PER_YEAR = 52

WINDOW = 3
N_SIM = 300
ETA = 0.5

# ==================== LOAD DATA ====================

def load_data():
    return {
        city: {
            ds: pd.read_csv(DATA_DIR / city / f"{ds}_{city}.csv")
            for ds in DATASETS
        }
        for city in CITIES
    }

# ==================== PROCESS DATA ====================

def process_data(data):
    processed = {}

    for city in CITIES:
        processor = DataProcessor(data[city]["R1"])

        processed[city] = {}
        processed[city]["R1"] = processor.linear_interpolation()

        for ds in ["R21", "R22", "R23"]:
            processed[city][ds] = processor.zero_imputation()

    return processed

# ==================== FILTER PERIOD ====================

def extract_period(data):
    start_week = (START_YEAR - BASE_YEAR) * WEEKS_PER_YEAR + 1

    return {
        city: {
            'cases': data[city]["R1"][data[city]["R1"]['epi_weeks'] >= start_week]['cases'].values,
            'temperature': data[city]["R1"][data[city]["R1"]['epi_weeks'] >= start_week]['avg_temp'].values,
            'humidity': data[city]["R1"][data[city]["R1"]['epi_weeks'] >= start_week]['avg_humid'].values,
        }
        for city in CITIES
    }

# ==================== STATIONARITY ====================

def run_stationarity_tests(cases):
    tester = StationarityTests()

    adf_results = []
    kpss_results = []

    for city in CITIES:
        adf_results.append(tester.adf(cases[city], city))
        kpss_results.append(tester.kpss(cases[city], city))

    return (
        pd.concat(adf_results, ignore_index=True),
        pd.concat(kpss_results, ignore_index=True)
    )

# ==================== TIME SERIES ANALYSIS ====================

def run_time_series_analysis(cases, temperatures, humidities):
    analysis = TimeSeriesAnalysis()

    START_WEEK = 106
    END_WEEK = 158

    results = {}

    for city in CITIES:
        cases_2024 = cases[city][START_WEEK:END_WEEK]
        temp_2024 = temperatures[city][START_WEEK:END_WEEK]
        humid_2024 = humidities[city][START_WEEK:END_WEEK]

        results[city] = {
            "stl": analysis.stl_decomposition(cases[city], 52),
            "acf": analysis.acf(cases_2024, 52),
            "pacf": analysis.pacf(cases_2024, 26),
            "ccf_temp": analysis.ccf(cases_2024, temp_2024, 12),
            "ccf_humid": analysis.ccf(cases_2024, humid_2024, 12),
        }

    return results

# ==================== MODELING ====================

def run_models(cases):
    models = ForecastModels()
    metrics = Metrics()

    results = []

    for city in CITIES:
        sma = models.SMA_model(cases[city], WINDOW, N_SIM)
        ema = models.EMA_model(cases[city], WINDOW, N_SIM, ETA)

        actual = sma["Actual"].values
        sma_pred = sma["Forecast"].values
        ema_pred = ema["Forecast"].values

        results.append({
            "city": city,
            "SMA_sMAPE": metrics.sMAPE(actual, sma_pred),
            "SMA_MAE": metrics.MAE(actual, sma_pred),
            "SMA_MSE": metrics.MSE(actual, sma_pred),
            "SMA_Theil": metrics.Utheil(actual, sma_pred),
            "EMA_sMAPE": metrics.sMAPE(actual, ema_pred),
            "EMA_MAE": metrics.MAE(actual, ema_pred),
            "EMA_MSE": metrics.MSE(actual, ema_pred),
            "EMA_Theil": metrics.Utheil(actual, ema_pred),
        })

    return pd.DataFrame(results)

# ==================== MAIN ====================

def main():

    data = load_data()
    data = process_data(data)
    data_period = extract_period(data)

    cases = {city: data_period[city]['cases'] for city in CITIES}
    temperatures = {city: data_period[city]['temperature'] for city in CITIES}
    humidities = {city: data_period[city]['humidity'] for city in CITIES}

    # ==================== STATIONARITY TESTS ====================
    # adf, kpss = run_stationarity_tests(cases)
    # print("\n=== ADF ===")
    # print(adf)
    # print("\n=== KPSS ===")
    # print(kpss)

    # ==================== TIME SERIES ANALYSIS ====================
    #ts_results = run_time_series_analysis(cases, temperatures, humidities)

    # ==================== MODELING ====================
    df_results = run_models(cases)

    print("\n=== MODEL RESULTS ===")
    print(df_results.round(2))


if __name__ == "__main__":
    main()