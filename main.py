"""
-----------------------------------------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------------------------------------
Title: Numerical exercise - Electricity & Magnetism, HUJI 2022

Copyright: Roy Guggenheim, Oren Gercenshtein

Description: - Solves the problem of a particle's position under constant electric and magnetic fields.
             - Utilizes the solution in order to implement a Wien-filter.
             - Graphs the process.
-----------------------------------------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------------------------------------
"""

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


def update_position(i, method):
    """
    Updates position vector of the charged particle.
    :param i: iteration we are on
    :return: nothing
    """
    if method == 'TAYLOR':
        r[i][Coordinate.X] = r[i - 1][Coordinate.X] + v[i - 1][Coordinate.X] * DELTA_T
        r[i][Coordinate.Y] = r[i - 1][Coordinate.Y] + v[i - 1][Coordinate.Y] * DELTA_T
        r[i][Coordinate.Z] = r[i - 1][Coordinate.Z] + v[i - 1][Coordinate.Z] * DELTA_T
    elif method == 'MIDPOINT':
        # TODO add method
        pass
    elif method == 'RUNGE-JUTTA':
        # TODO add method
        pass


def update_velocity(i, method):
    """
    Updates velocity vector of the charged particle.
    :param i: iteration we are on
    :return: nothing
    """
    if method == 'TAYLOR':
        factor = q / m
        accelerations = [0, factor * (E - B * v[i - 1][Coordinate.Z]), factor * B * v[i - 1][Coordinate.Y]]
        v[i][1] = v[i - 1][Coordinate.Y] + accelerations[Coordinate.Y] * DELTA_T
        v[i][2] = v[i - 1][Coordinate.Z] + accelerations[Coordinate.Z] * DELTA_T
    elif method == 'MIDPOINT':
        # TODO add method
        pass
    elif method == 'RUNGE-JUTTA':
        # TODO add method
        pass


def calculate(method):
    """
    Iterates until end of T, each iteration calculating the current positions and velocities.
    :return: nothing
    """
    for i in range(N - 1):
        update_position(i + 1, method)
        update_velocity(i + 1, method)


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


def reset():
    """
    Reset the position, velocity arrays for the use of other methods.
    :return:
    """
    global r, v
    r, v = np.zeros((N, 3)), np.zeros((N, 3))
    v[0][2] = 3 * E / B


for method in ["TAYLOR", "MIDPOINT", "RUNGE-JUTTA"]:
    calculate(method)
    graph_motion()
    graph_error()
    reset()
