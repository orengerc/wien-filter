"""
Holds equations used for fitting
"""

import scipy.optimize as opt
import numpy as np


class CurveFit:
    def __init__(self, x, y, fit_function):
        self._x = x
        self._y = y
        self._fit_func = fit_function
        self._fit_params, self._covariance = opt.curve_fit(self._fit_func, x, y)

    def get_fit_params(self):
        return self._fit_params

    def get_fit(self):
        x_fit = np.linspace(np.amin(self._x), np.amax(self._x), num=10000)
        y_fit = self._fit_func(x_fit, *self._fit_params)
        return x_fit, y_fit
