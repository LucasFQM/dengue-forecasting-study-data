import pandas as pd


class DataProcessor:
    """
    Process epidemiological datasets, including handling of missing values.
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initialize the processor with a dataset.

        Parameters
        ----------
        data : pandas.DataFrame
            Input dataset.
        """
        self.data = data.copy()

    def linear_interpolation(self) -> pd.DataFrame:
        """
        Interpolate missing values in meteorological variables.

        Returns
        -------
        pandas.DataFrame
            DataFrame with interpolated values.

        Raises
        ------
        ValueError
            If required columns are not present in the dataset.
        """

        cols = ['avg_temp', 'avg_humid']

        for col in cols:
            if col not in self.data.columns:
                raise ValueError(f"Column '{col}' not found in dataset.")
            self.data[col] = self.data[col].interpolate(method='linear')

        return self.data

    def zero_imputation(self) -> pd.DataFrame:
        """
        Replace missing values with zero.

        Returns
        -------
        pandas.DataFrame
            DataFrame with missing values replaced.
        """

        self.data = self.data.fillna(0)

        return self.data