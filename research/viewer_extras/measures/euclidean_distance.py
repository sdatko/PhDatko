#!/usr/bin/env python3
#
# This file intentionally duplicates some code to be self-contained.
# pylint: disable=duplicate-code (R0801)
#

import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import numpy as np


def euclidean_distance() -> plt.Figure:
    #
    # Data generation
    #
    np.random.seed(42)

    vx1, vy1 = (2.3, 3.0)  # ID
    vx2, vy2 = (3.6, 3.0)  # OOD

    mean = [2, 2]
    cov = [[0.2, -0.1],
           [-0.1, 0.5]]

    n = 100
    x, y = np.random.multivariate_normal(mean, cov, n).T

    mean_x = np.mean(x)
    mean_y = np.mean(y)

    #
    # Plot creation
    #
    fig, ax = plt.subplots(nrows=1, ncols=1)

    #
    # Draw cluster
    #
    ax.scatter(x, y,
               c='tab:blue', s=25, label=r'$x_i \in K$', alpha=0.5, zorder=2)
    ax.scatter(mean_x, mean_y,
               c='tab:orange', s=75, label=r'$\mu_{K}$', alpha=0.9, zorder=2)

    for xi, yi in zip(x, y):
        ax.plot((mean_x, xi), (mean_y, yi),
                linestyle='-', color='orange', alpha=0.15, zorder=1)

    #
    # Draw ID example
    #
    ax.plot((mean_x, vx1), (mean_y, vy1),
            color='tab:green', alpha=0.5, linestyle='-', zorder=3)
    ax.scatter(vx1, vy1,
               color='tab:green', s=75, label=r'$v_1$', alpha=0.9, zorder=4)

    #
    # Draw OOD example
    #
    ax.plot((mean_x, vx2), (mean_y, mean_y), 'r--', alpha=0.5, zorder=1)
    ax.plot((vx2, vx2), (mean_y, vy2), 'r--', alpha=0.5, zorder=1)
    ax.plot((mean_x, vx2), (mean_y, vy2),
            color='tab:red', alpha=0.5, linestyle='-', zorder=3)
    ax.scatter(vx2, vy2,
               color='tab:red', s=75, label=r'$v_2$', alpha=0.9, zorder=4)

    #
    # Draw fixed labels
    #
    ax.text(3.05, 1.67, 'a', c='red', fontsize=16)
    ax.text(3.64, 2.2, 'b', c='red', fontsize=16)
    ax.text(2.85, 2.60, r'$c = \sqrt{a^2 + b^2}$',
            rotation=17, c='red', fontsize=16)

    #
    # Plot settings
    #
    fig.set_figheight(4)
    fig.set_figwidth(9)

    ax.set_xlim(1.0, 4.0)
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
    FILENAME = 'euclidean-distance.pdf'
    figure = euclidean_distance()
    figure.savefig(FILENAME, dpi=300, bbox_inches='tight')
    print('Saved:', FILENAME)
