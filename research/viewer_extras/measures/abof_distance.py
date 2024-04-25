#!/usr/bin/env python3
#
# This file intentionally duplicates some code to be self-contained.
# pylint: disable=duplicate-code (R0801)
#

import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import numpy as np


def abof_distance() -> plt.Figure:
    #
    # Data generation
    #
    np.random.seed(42)

    vx1, vy1 = (2.0, 2.25)  # ID
    vx2, vy2 = (3.5, 2.5)  # corner-case
    vx3, vy3 = (5.8, 2.0)  # OOD

    mean = [2, 2]
    cov = [[1.0, -0.1],
           [-0.1, 0.6]]

    n = 20
    x, y = np.random.multivariate_normal(mean, cov, n).T

    #
    # Plot creation
    #
    fig, ax = plt.subplots(nrows=1, ncols=1)

    #
    # Draw cluster
    #
    ax.scatter(x, y,
               c='tab:blue', s=25, label=r'$x_i \in K$', alpha=0.5, zorder=2)

    #
    # Draw ID example
    #
    for xi, yi in zip(x, y):
        ax.plot((xi, vx1), (yi, vy1),
                color='tab:green', alpha=0.5, linestyle='-', zorder=3)

    ax.scatter(vx1, vy1,
               color='tab:green', s=75, label=r'$v_1$', alpha=0.9, zorder=4)

    #
    # Draw corner example
    #
    for xi, yi in zip(x, y):
        ax.plot((xi, vx2), (yi, vy2),
                color='tab:orange', alpha=0.4, linestyle='-', zorder=3)

    ax.scatter(vx2, vy2,
               color='tab:orange', s=75, label=r'$v_2$', alpha=0.9, zorder=4)

    #
    # Draw OOD example
    #
    for xi, yi in zip(x, y):
        ax.plot((xi, vx3), (yi, vy3),
                color='tab:red', alpha=0.1, linestyle='-', zorder=3)

    ax.scatter(vx3, vy3,
               color='tab:red', s=75, label=r'$v_3$', alpha=0.9, zorder=4)

    #
    # Plot settings
    #
    fig.set_figheight(3)
    fig.set_figwidth(9)

    ax.set_xlim(0.5, 6.5)
    ax.set_ylim(0.5, 3.5)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.legend(loc='lower right')

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
    FILENAME = 'abof-distance.pdf'
    figure = abof_distance()
    figure.savefig(FILENAME, dpi=300, bbox_inches='tight')
    print('Saved:', FILENAME)
