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
from openset.models import MinMaxOutFactor  # noqa: E402
from openset.models import MinMaxOutScore  # noqa: E402
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
    MinMaxOutFactor(),
    MinMaxOutScore(),
    SEuclidean(),
]

ITERATIONS = 5

N_CORRELATED = (
    Decimal('0.00'),
    Decimal('0.10'),
    Decimal('0.20'),
    Decimal('0.30'),
    Decimal('0.40'),
    Decimal('0.50'),
    Decimal('0.60'),
    Decimal('0.70'),
    Decimal('0.80'),
    Decimal('0.90'),
    Decimal('1.00'),
)

COVARIANCES = (
    Decimal('0.00'),
    Decimal('0.10'),
    Decimal('0.20'),
    Decimal('0.30'),
    Decimal('0.40'),
    Decimal('0.50'),
    Decimal('0.60'),
    Decimal('0.70'),
    Decimal('0.80'),
    Decimal('0.90'),
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
        N_CORRELATED,
        COVARIANCES,
        OUTLIERS_CORRELATED,
    )
    total = (
        len(DISTANCES)
        * len(MODELS)
        * ITERATIONS  # seeds
        * len(N_CORRELATED)
        * len(COVARIANCES)
        * len(OUTLIERS_CORRELATED)
    )

    runner = Runner()
    experiment = Correlations(cached=True)

    runner.run(experiment.get, iterator, unpack=True, length=total)


if __name__ == '__main__':
    main()
