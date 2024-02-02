#!/usr/bin/env python3

import os
import itertools

os.environ['MKL_NUM_THREADS'] = '2'
os.environ['NUMEXPR_NUM_THREADS'] = '2'
os.environ['OMP_NUM_THREADS'] = '2'

from openset.experiments.overlapping import BoundingBoxes  # noqa: E402
from openset.utils.runner import Runner  # noqa: E402


#
# Parameters for the experiment
#
DIMENSIONS = (
    1, 2, 3, 5,
    10, 25, 50, 75,
    100, 250, 500, 750,
    1000, 2500, 5000, 7500,
    10000,
)

DISTRIBUTIONS = (
    'correlated2525',
    'correlated5050',
    'correlated7575',
    'gaussian',
    'triangular',
    'uniform',
)

SAMPLES = (
    10, 25, 50, 75,
    100, 250, 500, 750,
    1000, 2500, 5000, 7500,
    10000, 25000, 50000,
)

ITERATIONS = 5


#
# Here we go!
#
def main():
    iterator = itertools.product(
        DIMENSIONS,
        DISTRIBUTIONS,
        SAMPLES,
        range(ITERATIONS),  # seeds
    )

    runner = Runner()
    experiment = BoundingBoxes(cached=True)

    runner.run(experiment.get, tuple(iterator), unpack=True)


if __name__ == '__main__':
    main()