# Modeling and Analytical Study of Dengue Dynamics in Brazilian Cities

This repository contains the code and datasets used to reproduce the study titled *"Modeling and Analytical Study of Dengue Dynamics in Brazilian Cities"*.

This work was developed at the Aeronautics Institute of Technology (ITA) by:

- **Author:** Lucas Ferreira Quintão Moreira  
- **Advisor:** Takashi Yoneyama  

---

## 📌 Overview

This project investigates the dynamics of dengue in three Brazilian cities:

- São José dos Campos  
- Resende  
- Ouro Preto  

The analysis integrates statistical and computational approaches, including:

- Data preprocessing  
- Stationarity testing (ADF and KPSS)  
- Time series decomposition (STL)  
- Correlation analysis (ACF, PACF, CCF)  
- Forecasting models based on moving averages combined with Monte Carlo simulation  

---

## 📁 Repository Structure

```
.
├── code/
│   ├── main.py
│   ├── models.py
│   ├── analysis.py
│   └── processing.py
├── data/
└── README.md
```

### 📂 `code/`
Contains all source code:

- `main.py` → Main execution script  
- `models.py` → Forecasting models and evaluation metrics  
- `analysis.py` → Statistical tests and time series analysis  
- `processing.py` → Data preprocessing  

### 📂 `data/`
Contains epidemiological and climate datasets used in the study.  

---

## ⚙️ Methodology

### Data Processing

Missing values are handled using:

- Linear interpolation (temperature and air humidity)  
- Zero imputation (other variables)  

---

### Stationarity Tests

- Augmented Dickey-Fuller (ADF)  
- Kwiatkowski-Phillips-Schmidt-Shin (KPSS)  

These tests provide complementary evidence regarding the stationarity properties of the time series.

---

### Time Series Analysis

- STL decomposition (trend, seasonality, and residual components)  
- Autocorrelation Function (ACF)  
- Partial Autocorrelation Function (PACF)  
- Cross-Correlation Function (CCF) between dengue cases and climate variables  

---

## 🤖 Forecasting Models

### 📊 Simple Moving Average (SMA)
- Uses the mean and standard deviation of a rolling window  
- Monte Carlo simulation is used to generate prediction intervals  

### 📊 Exponential Moving Average (EMA)
- Applies exponential smoothing to estimate expected values  
- Monte Carlo simulation is used to generate prediction intervals  

### 🎲 Monte Carlo Simulation
- Assumes a Student's \(t\)-distribution  
- Generates multiple simulated scenarios  
- Produces:
  - Point forecasts  
  - 95% prediction intervals  

---

## 📈 Evaluation Metrics

- sMAPE (Symmetric Mean Absolute Percentage Error)  
- MAE (Mean Absolute Error)  
- MSE (Mean Squared Error)  
- Theil's U statistic (comparison with a naive benchmark)  

---

## 🚀 How to Run

### 1. Navigate to the code directory

```bash
cd code
```

### 2. Run the main script

```bash
python main.py
```

### 3. Output

The script prints:

- Forecast performance metrics for each city  
- A comparison between SMA and EMA models  

---

## 📊 Example Output

=== SMA MODEL RESULTS ===

| City                 | sMAPE | MAE | MSE   | Theil | Coverage |
|----------------------|-------|-----|-------|-------|----------|
| Resende              | 25.43 | 12.5| 245.3 | 0.82  | 0.95     |
| São José dos Campos  | ...   | ... | ...   | ...   | ...      |
| Ouro Preto           | ...   | ... | ...   | ...   | ...      |

=== EMA MODEL RESULTS ===

| City                 | sMAPE | MAE | MSE   | Theil | Coverage |
|----------------------|-------|-----|-------|-------|----------|
| Resende              | 23.78 | 11.2| 198.7 | 0.79  | 0.94     |
| São José dos Campos  | ...   | ... | ...   | ...   | ...      |
| Ouro Preto           | ...   | ... | ...   | ...   | ...      |

## 🔬 Notes

- Forecasts are constrained to non-negative values  
- Results are rounded to integers to reflect count data  
- The standard deviation of the rolling window is used as a proxy for forecast uncertainty  

---

## 📎 Reproducibility

- A fixed random seed ensures reproducibility  
- All results can be replicated by executing `main.py`  

---

## 📄 License

This project is intended for academic and research purposes.

