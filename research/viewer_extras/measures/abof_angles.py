#!/usr/bin/env python3
#
# This file intentionally duplicates some code to be self-contained.
# pylint: disable=duplicate-code (R0801)
#

import math
from itertools import combinations

import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import numpy as np


def abof_score(vec: np.ndarray, data: np.ndarray) -> (list, float):
    angles = []
    vectors = data - vec

    for vec1, vec2 in combinations(vectors, 2):
        norm1 = np.sqrt(vec1.dot(vec1))
        norm2 = np.sqrt(vec2.dot(vec2))

        if norm1 == 0 or norm2 == 0:
            continue

        angles.append(
            math.degrees(
                math.acos(
                    vec1.dot(vec2) / norm1 / norm2
                )
            )
        )

    return angles, np.var(angles, axis=0)


def abof_angles() -> plt.Figure:
    #
    # Data generation
    #
    np.random.seed(42)

    vx1, vy1 = (2.0, 2.25)
    vx2, vy2 = (3.5, 2.5)
    vx3, vy3 = (5.8, 2.0)

    mean = [2, 2]
    cov = [[1.0, -0.1],
           [-0.1, 0.6]]

    n = 20
    data = np.random.multivariate_normal(mean, cov, n)

    #
    # Calculate angles and variances
    #
    angles1, var1 = abof_score(np.array([vx1, vy1]), data)
    angles2, var2 = abof_score(np.array([vx2, vy2]), data)
    angles3, var3 = abof_score(np.array([vx3, vy3]), data)

    xs = list(range(len(angles1)))

    #
    # Plot creation
    #
    fig, ax = plt.subplots(nrows=1, ncols=1)

    #
    # Draw ID example
    #
    ax.plot(xs, angles1,
            label=r'$Var_1 \approx $' + str(round(var1, 2)),
            color='tab:green', alpha=0.7, linestyle='-', linewidth=1.0)

    #
    # Draw corner example
    #
    ax.plot(xs, angles2,
            label=r'$Var_2 \approx $' + str(round(var2, 2)),
            color='tab:orange', alpha=0.8, linestyle='-', linewidth=1.25)

    #
    # Draw OOD example
    #
    ax.plot(xs, angles3,
            label=r'$Var_3 \approx $' + str(round(var3, 2)),
            color='tab:red', alpha=0.9, linestyle='-', linewidth=1.5)

    #
    # Plot settings
    #
    fig.set_figheight(3)
    fig.set_figwidth(9)

    ax.set_xlim(min(xs), max(xs))
    ax.set_ylim(0.0, 180.0)
    ax.xaxis.set_ticks(np.arange(0, 190, 15))
    ax.yaxis.set_ticks(np.arange(0, 181, 15))
    ax.set_xlabel('index of pair')
    ax.set_ylabel('angle [deg]')
    ax.legend(loc='upper right')

    ax.locator_params(nbins=10, axis='both')
    ax.xaxis.set_minor_locator(tck.AutoMinorLocator())
    ax.yaxis.set_minor_locator(tck.AutoMinorLocator())
    ax.grid(True, alpha=0.25)

    #
    # Result
    #
    return fig


#
# If the script is invoked directly...
#
if __name__ == '__main__':
    FILENAME = 'abof-angles.pdf'
    figure = abof_angles()
    figure.savefig(FILENAME, dpi=300, bbox_inches='tight')
    print('Saved:', FILENAME)
