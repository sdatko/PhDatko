#!/usr/bin/env python3

import os
import itertools

os.environ['MKL_NUM_THREADS'] = '2'
os.environ['NUMEXPR_NUM_THREADS'] = '2'
os.environ['OMP_NUM_THREADS'] = '2'

from openset.experiments.distributions import Generated  # noqa: E402
from openset.models import AngleBasedOutlierFactor  # noqa: E402
from openset.models import Euclidean  # noqa: E402
from openset.models import IntegratedRankWeightedDepth  # noqa: E402
from openset.models import KNearestNeighbors  # noqa: E402
from openset.models import LocalOutlierFactor  # noqa: E402
from openset.models import Mahalanobis  # noqa: E402
from openset.models import Manhattan  # noqa: E402
from openset.models import SEuclidean  # noqa: E402
from openset.utils.runner import Runner  # noqa: E402


#
# Parameters for the experiment shared by all instances
#
DISTANCES = (
    1,
    2,
    4,
    8,
    16,
)

DISTRIBUTIONS = (
    'gaussian',
    'triangular',
    'uniform',
)

ITERATIONS = 5


#
# Models that are fast and easy to calculate:
# – Euclidean,
# – KNearestNeighbors,
# – LocalOutlierFactor,
# – Manhattan,
# – MinMaxWindow,
# – SEuclidean,
#
dimensions = (
    1, 2, 3, 5,
    10, 25, 50, 75,
    100, 250, 500, 750,
    1000, 2500, 5000, 7500,
    10000,
)

training_samples = (
    25, 50, 75,
    100, 250, 500, 750,
    1000, 2500, 5000, 7500,
    10000, 25000, 50000,
)

models = [
    Euclidean(),
    KNearestNeighbors(5),
    KNearestNeighbors(10),
    KNearestNeighbors(20),
    LocalOutlierFactor(5),
    LocalOutlierFactor(10),
    LocalOutlierFactor(20),
    Manhattan(),
    SEuclidean(),
]

FAST_TO_CALCULATE_PARAMETERS = itertools.product(
    dimensions,
    DISTANCES,
    DISTRIBUTIONS,
    models,
    training_samples,
    range(ITERATIONS),  # seeds
)


#
# ABOF model
# – reduced dimensions and number of training samples
#
dimensions = (
    1, 2, 3, 5,
    10, 25, 50, 75,
    100, 250, 500, 750,
)

training_samples = (
    10, 25, 50, 75,
    100, 250, 500, 750,
    1000, 2500, 5000, 7500,
)

models = [
    AngleBasedOutlierFactor(),
]

ABOF_PARAMETERS = itertools.product(
    dimensions,
    DISTANCES,
    DISTRIBUTIONS,
    models,
    training_samples,
    range(ITERATIONS),
)


#
# IRWD model
# – reduced dimensions and number of training samples
#
dimensions = (
    1, 2, 3, 5,
    10, 25, 50, 75,
    100, 250, 500, 750,
    1000,
)

training_samples = (
    10, 25, 50, 75,
    100, 250, 500, 750,
    1000, 2500, 5000, 7500,
    10000,
)

models = [
    IntegratedRankWeightedDepth(100),
    IntegratedRankWeightedDepth(500),
    IntegratedRankWeightedDepth(1000),
]

IRWD_PARAMETERS = itertools.product(
    dimensions,
    DISTANCES,
    DISTRIBUTIONS,
    models,
    training_samples,
    range(ITERATIONS),
)


#
# Mahalanobis model
# – reduced dimensions as the default matrix decomposition takes a lot of time
#
dimensions = (
    1, 2, 3, 5,
    10, 25, 50, 75,
    100, 250, 500, 750,
    1000, 2500, 5000,
)

training_samples = (
    10, 25, 50, 75,
    100, 250, 500, 750,
    1000, 2500, 5000, 7500,
    10000, 25000, 50000,
)

models = [
    Mahalanobis(),
]

MAHALANOBIS_PARAMETERS = itertools.product(
    dimensions,
    DISTANCES,
    DISTRIBUTIONS,
    models,
    training_samples,
    range(ITERATIONS),
)


#
# Here we go!
#
def main():
    iterator = itertools.chain(
        FAST_TO_CALCULATE_PARAMETERS,
        ABOF_PARAMETERS,
        IRWD_PARAMETERS,
        MAHALANOBIS_PARAMETERS,
    )

    runner = Runner()
    experiment = Generated(cached=True)

    runner.run(experiment.get, tuple(iterator), unpack=True)


if __name__ == '__main__':
    main()
