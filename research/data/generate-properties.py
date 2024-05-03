#!/usr/bin/env python3

from decimal import Decimal
import os
import itertools

os.environ['MKL_NUM_THREADS'] = '2'
os.environ['NUMEXPR_NUM_THREADS'] = '2'
os.environ['OMP_NUM_THREADS'] = '2'

from openset.experiments.properties import MVNEstimation  # noqa: E402
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

SAMPLES = (
    10, 25, 50, 75,
    100, 250, 500, 750,
    1000, 2500, 5000, 7500,
    10000, 25000, 50000,
)

N_CORRELATED = (
    Decimal('0.00'),
    Decimal('0.20'),
    Decimal('0.40'),
    Decimal('0.60'),
    Decimal('0.80'),
    Decimal('1.00'),
)

COVARIANCES = (
    Decimal('0.00'),
    Decimal('0.20'),
    Decimal('0.40'),
    Decimal('0.60'),
    Decimal('0.80'),
    Decimal('1.00'),
)

ITERATIONS = 5


#
# Here we go!
#
def main():
    iterator = itertools.product(
        DIMENSIONS,
        SAMPLES,
        N_CORRELATED,
        COVARIANCES,
        range(ITERATIONS),  # seeds
    )
    total = (
        len(DIMENSIONS)
        * len(SAMPLES)
        * len(N_CORRELATED)
        * len(COVARIANCES)
        * ITERATIONS  # seeds
    )

    runner = Runner()
    experiment = MVNEstimation(cached=True)

    runner.run(experiment.get, iterator, unpack=True, length=total)


if __name__ == '__main__':
    main()
