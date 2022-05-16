"""
Hold pandas dataframe of given excel sheet
Performs various read operations which all return numpy arrays
"""

import numpy as np
import pandas as pd


def clean_vector(x):
    return x[~np.isnan(x)]


class DataHandler:
    def __init__(self, filepath):
        self._path = filepath
        self._df = pd.read_excel(self._path)

    def get_columns(self, column_names):
        res = []
        for col_name in column_names:
            col = clean_vector(self._df[col_name].to_numpy())
            res.append(col)
        return tuple(res)

    def add_column_to_excel(self, name, values):
        self._df[name] = pd.Series(values)
        self._df.to_excel(self._path)
        self._df = pd.read_excel(self._path)

    def drop_first_n_rows(self, n):
        self._df = self._df.iloc[n+1:, :]

    def save_df_back_to_excel(self):
        self._df = pd.read_excel(self._path)
