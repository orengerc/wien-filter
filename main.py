"""
Runs analysis according to specific demands
"""

from DataHandler import *
from Graph import *
from CurveFit import *
from Equations import *
from PIL import Image
from ImageHandler import *
import numpy as np
from scipy.stats import linregress
import os
import matplotlib.pyplot as plt
import csv
import pandas as pd
from scipy.ndimage import gaussian_filter
import cv2


def parse_txt_to_dataframe(path):
    df = pd.DataFrame()
    with open(path) as tsv:
        for i, line in enumerate(csv.reader(tsv, delimiter="\t")):
            if i > 7 and i % 13 == 0:
                df = df.append([line])
    df = df.astype(float)
    df.drop(df.columns[[0, 1]], axis=1, inplace=True)
    df.iloc[:, 0] = df.iloc[:, 0].map(lambda x: -x)
    df.iloc[:, 1] = df.iloc[:, 1].map(lambda x: (x + 0.05) / 0.107)
    return df


def correct(df):
    mat = df.to_numpy()
    corr = correction_data.to_numpy()
    if df.shape[0] < correction_data.shape[0]:
        corr = corr[:df.shape[0]]
    else:
        mat = mat[:corr.shape[0]]
    mat[:, 0] = np.subtract(mat[:, 0], corr[:, 0])
    return pd.DataFrame(mat)


def graph_interference_pattern(df, kind):
    print("graphing")
    g = Graph(df.iloc[:, 1], df.iloc[:, 0])
    g.set_labels("Light Intensity vs. Angle, {}".format(filename), "Angle (deg)", "Intensity (AU)")
    if kind == "1_slit":
        g.plot_with_fit(one_slit)
    elif kind == "2_slits":
        pass
        # g.plot_with_fit(two_slits)
    print("fit parameters for {}:\n".format(filename), g.get_fit_parameters())


def write_df_to_csv(df, name):
    print("to csv")
    df.to_csv(os.path.join(folder, name) + '.csv')


if __name__ == '__main__':
    correction_data = parse_txt_to_dataframe("data/week2/correction.txt")
    correction_data.to_csv("data/week2/correction_data.csv")

    for n_slits in ["2_slits"]:
        folder = "data\\week2\\{0}".format(n_slits)
        for filename in os.listdir(folder):
            print(filename)
            data_frame = parse_txt_to_dataframe(os.path.join(folder, filename))
            data_frame = correct(data_frame)
            write_df_to_csv(data_frame, os.path.splitext(filename)[0])
            graph_interference_pattern(data_frame, n_slits)

    phi = np.array([[1, 2, 3, 4, 5], [1, 2, 35, 5, 5], [1, 2, 3, 6, 7]])
    kernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    print(cv2.filter2D(phi, -1, kernel))
