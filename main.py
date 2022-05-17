"""
Numerical exercise
Roy & Oren
"""

import numpy as np
import matplotlib.pyplot as plt
from enum import IntEnum


class Coordinate(IntEnum):
    """
    Defines the coordinates used in the excercise.
    """
    X = 0
    Y = 1
    Z = 2


# Constants of the problem
E = 1
B = 1
q = 1
m = 1

# initialize the variables
omega = q * B / m
T = 2 * np.pi / omega
DELTA_T = 0.01
N = int(T / DELTA_T)

r, v = np.zeros((N, 3)), np.zeros((N, 3))
v[0][2] = 3 * E / B


def update_position(i):
    """
    Updates position vector of the charged particle.
    :param i: iteration we are on
    :return: nothing
    """
    r[i][Coordinate.X] = r[i - 1][Coordinate.X] + v[i - 1][Coordinate.X] * DELTA_T
    r[i][Coordinate.Y] = r[i - 1][Coordinate.Y] + v[i - 1][Coordinate.Y] * DELTA_T
    r[i][Coordinate.Z] = r[i - 1][Coordinate.Z] + v[i - 1][Coordinate.Z] * DELTA_T


def update_velocity(i):
    """
    Updates velocity vector of the charged particle.
    :param i: iteration we are on
    :return: nothing
    """
    factor = q / m
    accelerations = [0, factor * (E - B * v[i - 1][Coordinate.Z]), factor * B * v[i - 1][Coordinate.Y]]
    v[i][1] = v[i - 1][Coordinate.Y] + accelerations[Coordinate.Y] * DELTA_T
    v[i][2] = v[i - 1][Coordinate.Z] + accelerations[Coordinate.Z] * DELTA_T


def calculate():
    """
    Iterates until end of T, each iteration calculating the current positions and velocities.
    :return: nothing
    """
    for i in range(N - 1):
        update_position(i + 1)
        update_velocity(i + 1)


def graph_2D():
    """
    Plots position and velocity of particle in the Y axis vs. Z axis
    :return: nothing
    """
    plt.plot(r[:, Coordinate.Y], r[:, Coordinate.Z])
    plt.show()


def graph_3D():
    """
    Plots position and velocity of particle in the Y axis vs. Z axis
    :return: nothing
    """
    ax = plt.axes(projection='3d')
    ax.scatter3D(r[:, Coordinate.X], r[:, Coordinate.Y], r[:, Coordinate.Z], c=r[:, Coordinate.Z], cmap='Greens')
    plt.show()


def taylor_method():
    """

    :return:
    """
    calculate()
    graph_2D()
    graph_3D()


taylor_method()
