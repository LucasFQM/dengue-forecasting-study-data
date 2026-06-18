import matplotlib.pyplot as plt
import numpy as np


class ForecastPlots:
    """
    Visualization utilities for forecasting models.
    """

    def forecast_plot(self, results_df, city, model):

        x = np.arange(len(results_df))

        plt.figure(figsize=(12, 6))

        plt.plot(
            x,
            results_df["Actual"],
            label="Actual"
        )

        plt.plot(
            x,
            results_df["Forecast"],
            linestyle="-",
            label="Forecast",
            color='red'
        )

        plt.fill_between(
            x,
            results_df["Lower_95"],
            results_df["Upper_95"],
            alpha=0.2,
            label="95% PI"
        )

        plt.title(f"{city} ({model})")
        plt.xlabel("Epidemiological Weeks")
        plt.ylabel("Cases")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()