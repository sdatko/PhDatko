#!/usr/bin/env python3
#
# This file intentionally duplicates some code to be self-contained.
# pylint: disable=duplicate-code (R0801)
#
# The variable names like M, U, M_v indicate the matrices.
# pylint: disable=invalid-name (C0103)
#

import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import numpy as np


def irwd_fit(data: np.ndarray, n_proj: int = 1) -> (np.ndarray, np.ndarray):
    dimension = data.shape[1]

    mu = np.zeros(dimension)
    cov = np.identity(dimension)

    rng = np.random.default_rng(42)
    U = rng.multivariate_normal(mu, cov, n_proj).T

    U = U / np.linalg.norm(U, axis=0)  # normalized
    M = np.dot(data, U)

    return M, U


def irwd_score(vec: np.ndarray, M: np.ndarray, U: np.ndarray) -> float:
    v = np.dot(vec, U)
    M_v = M - v

    training_samples = M_v.shape[0]
    n_proj = M_v.shape[1]

    d_irw = sum(min((column <= 0).sum(), (column > 0).sum())
                for column in M_v.T) / training_samples / n_proj

    return d_irw


def irwd_issue() -> plt.Figure:
    #
    # Data generation
    #
    np.random.seed(42)

    xmin = 0.0
    xmax = 6.0
    ymin = 0.0
    ymax = 4.0
    nx = 60
    ny = 30

    vx1, vy1 = (3.6, 2.25)  # ID
    vx2, vy2 = (1.0, 1.5)  # OOD

    mean = [3, 2.25]
    cov = [[0.1, 0.1],
           [0.1, 1.0]]

    n = 50
    data = np.random.multivariate_normal(mean, cov, n)
    M, U = irwd_fit(data)

    xlist = np.linspace(xmin, xmax, nx)
    ylist = np.linspace(ymin, ymax, ny)

    x, y = np.meshgrid(xlist, ylist)
    z = [[irwd_score(np.array([xi, yi]), M, U) for xi in xlist]
         for yi in ylist]

    z = z / np.max(z)  # normalized

    #
    # Plot creation
    #
    fig, ax = plt.subplots(nrows=1, ncols=1)

    #
    # Draw cluster
    #
    ax.scatter(data[:, 0], data[:, 1],
               c='tab:blue', s=25, label=r'$x_i \in K$', alpha=0.75, zorder=3)

    #
    # Draw depths
    #
    ax.contour(x, y, z,
               alpha=0.25, colors='purple', linewidths=1, levels=20, zorder=1)
    cp = ax.contourf(x, y, z,
                     alpha=0.4, cmap='Purples', levels=20, zorder=1)
    cb = fig.colorbar(cp)
    cb.set_label('depth', labelpad=7)
    cb.ax.set_yticks(np.arange(0.0, 1.01, 0.1))

    #
    # Draw ID example
    #
    ax.scatter(vx1, vy1,
               color='tab:green', s=75, label=r'$v_1$', alpha=0.9, zorder=4)

    #
    # Draw OOD example
    #
    ax.scatter(vx2, vy2,
               color='tab:red', s=75, label=r'$v_2$', alpha=0.9, zorder=4)

    #
    # Draw projection vectors
    #
    origin = np.array([vx1, vy1])
    for xi, yi in U.T:
        ax.quiver(*origin, xi, yi,
                  color='orange', width=0.03, alpha=0.75,
                  angles='xy', units='xy', scale_units='xy', scale=1)

    origin = np.array([vx2, vy2])
    for xi, yi in U.T:
        ax.quiver(*origin, xi, yi,
                  color='orange', width=0.03, alpha=0.75,
                  angles='xy', units='xy', scale_units='xy', scale=1)

    ax.scatter([], [],
               marker=r'$\longrightarrow$',
               label=r'$u_i \in U$',
               color='orange', alpha=0.75, s=150)

    #
    # Plot settings
    #
    fig.set_figheight(3)
    fig.set_figwidth(9)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
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
    FILENAME = 'irwd-issue.pdf'
    figure = irwd_issue()
    figure.savefig(FILENAME, dpi=300, bbox_inches='tight')
    print('Saved:', FILENAME)
