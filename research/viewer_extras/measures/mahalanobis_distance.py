#!/usr/bin/env python3
#
# This file intentionally duplicates some code to be self-contained.
# pylint: disable=duplicate-code (R0801)
#

import matplotlib.pyplot as plt
import matplotlib.ticker as tck
from matplotlib import transforms
from matplotlib.patches import Ellipse
import numpy as np


def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    # Source: (01.03.2024)
    # https://matplotlib.org/stable/gallery/statistics/confidence_ellipse.html
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])

    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def mahalanobis_distance() -> plt.Figure:
    #
    # Data generation
    #
    np.random.seed(42)

    vx1, vy1 = (3.6, 3.0)  # ID
    vx2, vy2 = (3.0, 0.5)  # OOD

    mean = [2, 2]
    cov = [[1.6, 0.75],
           [0.75, 0.6]]

    n = 250
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
                linestyle='-', color='orange', alpha=0.1, zorder=1)

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
    ax.plot((mean_x, vx2), (mean_y, vy2),
            color='tab:red', alpha=0.5, linestyle='-', zorder=3)
    ax.scatter(vx2, vy2,
               color='tab:red', s=75, label=r'$v_2$', alpha=0.9, zorder=4)

    #
    # Draw distribution
    #
    confidence_ellipse(x, y, ax, n_std=1, facecolor=(0, 0, 1, 0.02),
                       label=r'$1\sigma$', edgecolor='blue', linestyle='-')
    confidence_ellipse(x, y, ax, n_std=2, facecolor=(0, 0, 1, 0.02),
                       label=r'$2\sigma$', edgecolor='blue', linestyle='--')
    confidence_ellipse(x, y, ax, n_std=3, facecolor=(0, 0, 1, 0.02),
                       label=r'$3\sigma$', edgecolor='blue', linestyle=':')

    #
    # Plot settings
    #
    fig.set_figheight(5)
    fig.set_figwidth(9)

    ax.set_xlim(-1.0, 6.0)
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
    FILENAME = 'mahalanobis-distance.pdf'
    figure = mahalanobis_distance()
    figure.savefig(FILENAME, dpi=300, bbox_inches='tight')
    print('Saved:', FILENAME)
