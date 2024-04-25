#!/usr/bin/env python3
#
# This file intentionally duplicates some code to be self-contained.
# pylint: disable=duplicate-code (R0801)
#

import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import numpy as np
from scipy.spatial import KDTree


def lof_distance() -> plt.Figure:
    #
    # Data generation
    #
    np.random.seed(42)

    vx1, vy1 = (1.5, 1.25)  # ID
    vx2, vy2 = (7.5, 2.5)  # OOD
    n_neighbors = 3

    mean = [3.5, 2.0]
    cov = [[2.0, -0.75],
           [-0.75, 1.0]]

    n = 36
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

    #
    # Draw ID example
    #
    distances, indices = kdtree.query((vx1, vy1), k=3)

    for index in indices:
        ax.plot((x[index], vx1), (y[index], vy1),
                color='tab:green', alpha=0.5, linestyle='-', zorder=3)

    circle = plt.Circle((vx1, vy1), max(distances),
                        color='tab:green', alpha=0.5, linestyle='--',
                        fill=False, clip_on=True)
    ax.add_patch(circle)

    ax.scatter(vx1, vy1,
               color='tab:green', s=75, label=r'$v_1$', alpha=0.9, zorder=4)

    #
    # Draw ID neighbors
    #
    for index in indices:
        nx = x[index]
        ny = y[index]

        # NOTE: The first point closest to (nx, ny) is itself, so we query n+1
        ndistances, nindices = kdtree.query((nx, ny), k=n_neighbors + 1)

        for nindex in nindices:
            ax.plot((x[nindex], nx), (y[nindex], ny),
                    color='tab:blue', alpha=0.25, linestyle='-', zorder=3)

        circle = plt.Circle((nx, ny), max(ndistances),
                            color='tab:blue', alpha=0.25, linestyle='--',
                            fill=False, clip_on=True)
        ax.add_patch(circle)

    #
    # Draw OOD example
    #
    distances, indices = kdtree.query((vx2, vy2), k=3)

    for index in indices:
        ax.plot((x[index], vx2), (y[index], vy2),
                color='tab:red', alpha=0.5, linestyle='-', zorder=3)

    circle = plt.Circle((vx2, vy2), max(distances),
                        color='tab:red', alpha=0.5, linestyle='--',
                        fill=False, clip_on=True)
    ax.add_patch(circle)

    ax.scatter(vx2, vy2,
               color='tab:red', s=75, label=r'$v_2$', alpha=0.9, zorder=4)

    #
    # Draw OOD neighbors
    #
    for index in indices:
        nx = x[index]
        ny = y[index]

        # NOTE: The first point closest to (nx, ny) is itself, so we query n+1
        ndistances, nindices = kdtree.query((nx, ny), k=n_neighbors + 1)

        for nindex in nindices:
            ax.plot((x[nindex], nx), (y[nindex], ny),
                    color='tab:blue', alpha=0.25, linestyle='-', zorder=3)

        circle = plt.Circle((nx, ny), max(ndistances),
                            color='tab:blue', alpha=0.25, linestyle='--',
                            fill=False, clip_on=True)
        ax.add_patch(circle)

    #
    # Plot settings
    #
    fig.set_figheight(4)
    fig.set_figwidth(9)

    ax.set_xlim(0.0, 9.0)
    ax.set_ylim(0.0, 4.0)
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
    FILENAME = 'lof-distance.pdf'
    figure = lof_distance()
    figure.savefig(FILENAME, dpi=300, bbox_inches='tight')
    print('Saved:', FILENAME)
