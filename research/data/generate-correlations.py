#!/usr/bin/env python3

from decimal import Decimal
import os
import itertools

os.environ['MKL_NUM_THREADS'] = '2'
os.environ['NUMEXPR_NUM_THREADS'] = '2'
os.environ['OMP_NUM_THREADS'] = '2'

from openset.experiments.correlations import Correlations  # noqa: E402
from openset.models import Euclidean  # noqa: E402
from openset.models import IntegratedRankWeightedDepth  # noqa: E402
from openset.models import KNearestNeighbors  # noqa: E402
from openset.models import LocalOutlierFactor  # noqa: E402
from openset.models import Mahalanobis  # noqa: E402
from openset.models import Manhattan  # noqa: E402
from openset.models import SEuclidean  # noqa: E402
from openset.utils.runner import Runner  # noqa: E402


#
# Parameters for the experiment
#
DISTANCES = (
    8,
    16,
)

MODELS = [
    Euclidean(),
    IntegratedRankWeightedDepth(100),
    IntegratedRankWeightedDepth(500),
    IntegratedRankWeightedDepth(1000),
    KNearestNeighbors(5),
    KNearestNeighbors(10),
    KNearestNeighbors(20),
    LocalOutlierFactor(5),
    LocalOutlierFactor(10),
    LocalOutlierFactor(20),
    Mahalanobis(),
    Manhattan(),
    SEuclidean(),
]

ITERATIONS = 5

N_FEATURES = (
    Decimal('0.00'),
    Decimal('0.20'),
    Decimal('0.40'),
    Decimal('0.60'),
    Decimal('0.80'),
    Decimal('1.00'),
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

OUTLIERS_CORRELATED = (
    True,
    False,
)


#
# Here we go!
#
def main():
    iterator = itertools.product(
        DISTANCES,
        MODELS,
        range(ITERATIONS),  # seeds
        N_FEATURES,
        N_CORRELATED,
        COVARIANCES,
        OUTLIERS_CORRELATED,
    )

    runner = Runner()
    experiment = Correlations(cached=True)

    runner.run(experiment.get, tuple(iterator), unpack=True)


if __name__ == '__main__':
    main()
