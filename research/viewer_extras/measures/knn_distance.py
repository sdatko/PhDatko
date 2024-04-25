#!/usr/bin/env python3
#
# This file intentionally duplicates some code to be self-contained.
# pylint: disable=duplicate-code (R0801)
#

import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import numpy as np
from scipy.spatial import KDTree


def knn_distance() -> plt.Figure:
    #
    # Data generation
    #
    np.random.seed(42)

    vx1, vy1 = (1.9, 3.8)  # ID
    vx2, vy2 = (4.8, 2.8)  # OOD

    mean = [2.0, 2.0]
    cov = [[0.7, 0.1],
           [0.1, 0.7]]

    n = 100
    x, y = np.random.multivariate_normal(mean, cov, n).T
    kdtree = KDTree(np.array(list(zip(x, y))))

    #
    # Plot creation
    #
    fig, ax = plt.subplots(nrows=1, ncols=1)

    #
    # Draw cluster
    #
    ax.scatter(x, y,
               c='tab:blue', s=25, label=r'$x_i \in K$', alpha=0.5, zorder=2)

    for xi, yi in zip(x, y):
        _, indices = kdtree.query((xi, yi), k=5)

        for index in indices:
            ax.plot((x[index], xi), (y[index], yi),
                    color='tab:blue', linestyle='-', alpha=0.1, zorder=1)

    #
    # Draw ID example
    #
    _, indices = kdtree.query((vx1, vy1), k=5)

    for index in indices:
        ax.plot((x[index], vx1), (y[index], vy1),
                color='tab:green', alpha=0.5, linestyle='-', zorder=3)

    ax.scatter(vx1, vy1,
               color='tab:green', s=75, label=r'$v_1$', alpha=0.9, zorder=4)

    #
    # Draw OOD example
    #
    _, indices = kdtree.query((vx2, vy2), k=5)

    for index in indices:
        ax.plot((x[index], vx2), (y[index], vy2),
                color='tab:red', alpha=0.5, linestyle='-', zorder=3)

    ax.scatter(vx2, vy2,
               color='tab:red', s=75, label=r'$v_2$', alpha=0.9, zorder=4)

    #
    # Plot settings
    #
    fig.set_figheight(4)
    fig.set_figwidth(9)

    ax.set_xlim(0.5, 5.0)
    ax.set_ylim(0.5, 4.5)
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
    FILENAME = 'knn-distance.pdf'
    figure = knn_distance()
    figure.savefig(FILENAME, dpi=300, bbox_inches='tight')
    print('Saved:', FILENAME)
