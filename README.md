# Dengue Forecasting and Time Series Analysis

This repository provides Python code for preprocessing, analyzing, and forecasting dengue time series using statistical methods and Monte Carlo simulation.

## Authors

- **Lucas Ferreira Quintão Moreira**
- **Takashi Yoneyama**

---

## Overview

This project investigates dengue dynamics and short-term forecasting in three Brazilian municipalities:

* São José dos Campos
* Resende
* Ouro Preto

The analysis combines statistical time series methods and computational approaches, including:

* Data preprocessing
* Stationarity testing (ADF and KPSS)
* Time series decomposition (STL)
* Partial autocorrelation analysis (PACF)
* Forecasting using Simple Moving Average (SMA) and Exponential Moving Average (EMA)
* Monte Carlo simulation based on the Student's t-distribution for uncertainty quantification

---

## Repository Structure

```text
.
├── Codes/
│   ├── main.py
│   ├── models.py
│   └── analysis.py
├── Data/
│   ├── Ouro Preto/
│   │   └── InfoDengue.csv
│   ├── Resende/
│   │   └── InfoDengue.csv
│   └── São José dos Campos/
│       └── InfoDengue.csv
├── .gitignore
├── README.md
└── requirements.txt
```

### `Codes/`

Contains the Python scripts used in the analysis and forecasting framework:

- `main.py`: Main script for executing the forecasting workflow.
- `models.py`: Implementation of the SMA and EMA forecasting models, Monte Carlo simulation, and evaluation metrics.
- `analysis.py`: Statistical analysis and time series diagnostic methods.

### `Data/`

Contains the weekly dengue case datasets obtained from InfoDengue for the three municipalities included in the study. Each municipality has a separate directory containing its corresponding `InfoDengue.csv` file.

---

## Methodology

### Time Series Analysis

- Stationarity tests (ADF and KPSS).
- STL decomposition.
- Partial Autocorrelation Function (PACF).

### Forecasting Models

- Simple Moving Average (SMA).
- Exponential Moving Average (EMA).
- Monte Carlo simulation using the Student's t-distribution to estimate predictive uncertainty.

The simulation generates nominal 95% prediction intervals based on local variability estimated from a rolling window.

### Model Evaluation

Forecasting performance is assessed using:

- MAE and MSE.
- Theil's U statistic.
- Prediction Interval Coverage Probability (PICP).
- Mean Prediction Interval Width (MPIW).
- Diebold-Mariano test for pairwise comparison of SMA and EMA forecasts.

---

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the main script

```bash
cd Codes
python main.py
```

### 3. Execution

The main script:

- Loads weekly dengue case data from InfoDengue.
- Generates SMA and EMA forecasts for each municipality.
- Calculates forecasting performance metrics.
- Performs pairwise Diebold-Mariano tests between SMA and EMA for each municipality.

The script displays the Diebold-Mariano test results in the console.

---

## Experimental Configuration and Reproducibility

The main experimental parameters are defined in `main.py`. The current default configuration is:

| Parameter | Value |
|---|---|
| Data source | InfoDengue |
| Municipalities | São José dos Campos, Resende, Ouro Preto |
| Study period | 2022–2025 |
| Base year | 2014 |
| Weeks per year | 52 |
| Forecasting models | SMA and EMA |
| Rolling window | 3 |
| EMA smoothing parameter (α) | 0.5 |
| Student's t degrees of freedom (ν) | 3 |
| Monte Carlo simulations | 300 |
| Random seed | 42 |
| Lower prediction percentile | 2.5 |
| Upper prediction percentile | 97.5 |
| STL period | 52 |
| STL robust estimation | True |
| PACF maximum lags | 26 |
| PACF method | Yule-Walker |
| ADF regression | Constant |
| KPSS regression | Constant |
| Theil's U epsilon | 1e-6 |


### Configuration of Manuscript Results

The following table specifies the forecasting and simulation parameters adjusted across the experiments reported in the manuscript. All other parameters remained at their default values defined in `main.py`.

| Parameter | Table 3 | Table 4 | Table 5 | Table 6 | Figure 5 | Figure 6 | Figure 7 | Figure 8 | Figure 9 | Figure 10 | Figure 11 | Figure 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Forecasting models | SMA | SMA | EMA | EMA | SMA | EMA | SMA | SMA | SMA | EMA | EMA | EMA |
| Rolling window | 3 and 5 | 3 | 3 | 3 | 3 | 3 | 3 and 5 | 3 and 5 | 3 and 5 | 3 | 3 | 3 |
| EMA smoothing parameter (α) | None | None | 0.2, 0.5 and 0.7 | 0.5 | None | 0.5 | None | None | None | 0.2, 0.5 and 0.7 | 0.2, 0.5 and 0.7 | 0.2, 0.5 and 0.7 |
| Student's t degrees of freedom (ν) | 3 | 1, 2 and 3 | 3 | 1, 2 and 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 |
| Monte Carlo simulations | 300 | 300 | 300 | 300 | 300 | 300 | 0 to 1000 | 0 to 1000 | 0 to 1000 | 0 to 1000 | 0 to 1000 | 0 to 1000 |

---

## Implementation Notes

- The Monte Carlo simulation uses the Student's t-distribution.
- The local scale parameter is estimated as `np.std(train_window)`, using the same rolling window as the forecasting model and NumPy's default standard deviation convention (`ddof=0`).
- The estimated local standard deviation acts as a scale parameter for the Student's t-distributed random variates.
- Negative simulated values are set to zero.
- The nominal 95% prediction intervals may exhibit different empirical coverage levels depending on the degrees of freedom.
- The Diebold-Mariano test compares SMA, EMA, and Naive forecasts separately for each municipality.

---

## License

This project is intended for academic and research purposes.

