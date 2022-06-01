"""
Numerical exercise
Roy & Oren
"""

import numpy as np
import matplotlib.pyplot as plt
from enum import IntEnum
from Graph import *


class Coordinate(IntEnum):
    """
    Defines the coordinates used in the excercise.
    """
    X = 0
    Y = 1
    Z = 2


# Constants of the problem
E = 5
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


def update_position(i, method):
    """
    Updates position vector of the charged particle.
    :param i: iteration we are on
    :return: nothing
    """
    global v_half

    if method == 'TAYLOR':
        r[i, :] = r[i - 1, :] + v[i - 1, :] * DELTA_T
    elif method == 'MIDPOINT':

        r[i, :] = r[i - 1, :] + v[i - 1, :] * DELTA_T
        k1 = DELTA_T * v[i - 1, :]

        r_half = r[i - 1] + 0.5 * k1
        v_half = v_half  # TODO make it right, using r_half

        k2 = DELTA_T * v_half

        # v_n+1 = v_n + k2
        r[i, :] = r[i - 1, :] + k2
    elif method == 'RUNGE-JUTTA':
        pass


def calculate_accelerations(velocities):
    factor = q / m
    return np.array(
        [0, factor * (E - B * velocities[Coordinate.Z]),
         factor * B * velocities[Coordinate.Y]])


v_half = np.zeros(3)


def update_velocity(i, method):
    """
    Updates velocity vector of the charged particle.
    :param i: iteration we are on
    :return: nothing
    """

    global v_half

    accelerations = calculate_accelerations(v[i - 1, :])
    if method == 'TAYLOR':
        v[i, :] = v[i - 1, :] + accelerations * DELTA_T
    elif method == 'MIDPOINT':
        k1 = DELTA_T * accelerations

        v_half = v[i - 1, :] + 0.5 * k1
        a_half = calculate_accelerations(v_half)

        k2 = DELTA_T * a_half

        # v_n+1 = v_n + k2
        v[i, :] = v[i - 1, :] + k2

    elif method == 'RUNGE-JUTTA':
        pass


def calculate(method):
    """
    Iterates until end of T, each iteration calculating the current positions and velocities.
    :return: nothing
    """
    for i in range(N - 1):
        update_velocity(i + 1, method)
        update_position(i + 1, method)


def graph_motion():
    """
    Plots position and velocity of particle in the Y axis vs. Z axis
    :return: nothing
    """
    # graph position of particle in Z,Y plane
    g = Graph(r[:, Coordinate.Y], r[:, Coordinate.Z])
    g.set_labels("Position of particle in Z,Y plane", "Y [Arbitrary Units]", "Z [Arbitrary Units]")
    g.plot()

    # graph position of particle in 3D
    ax = plt.axes(projection='3d')
    ax.scatter3D(r[:, Coordinate.X], r[:, Coordinate.Y], r[:, Coordinate.Z], c=r[:, Coordinate.Z], cmap='Greens')
    plt.show()

    # graph velocity of particle in Z,Y plane
    g = Graph(v[:, Coordinate.Y], v[:, Coordinate.Z])
    g.set_labels("Velocity of particle in Z,Y plane", "Velocity in Y direction [Arbitrary Units]",
                 "Velocity in Z direction [Arbitrary Units]")
    g.plot()


def graph_error():
    pass


def run(method):
    """
    Calculates the positions and velocities at any time using the desired method,
    then graphs the positions and velocities and their accumulated error.
    :return:
    """
    calculate(method)
    graph_motion()
    graph_error()


def reset():
    global r, v
    r, v = np.zeros((N, 3)), np.zeros((N, 3))


for method in ["TAYLOR", "MIDPOINT", "RUNGE-JUTTA"]:
    run(method)
    reset()
