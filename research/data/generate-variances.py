#!/usr/bin/env python3

from decimal import Decimal
import os
import itertools

os.environ['MKL_NUM_THREADS'] = '2'
os.environ['NUMEXPR_NUM_THREADS'] = '2'
os.environ['OMP_NUM_THREADS'] = '2'

from openset.experiments.variances import Variances  # noqa: E402
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

N_VARIED = (
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

VARIANCES = (
    Decimal('0.25'),
    Decimal('0.50'),
    Decimal('0.75'),
    Decimal('1.00'),
    Decimal('1.25'),
    Decimal('1.50'),
    Decimal('1.75'),
    Decimal('2.00'),
    Decimal('2.25'),
    Decimal('2.50'),
    Decimal('2.75'),
    Decimal('3.00'),
    Decimal('3.25'),
    Decimal('3.50'),
    Decimal('3.75'),
    Decimal('4.00'),
)

OUTLIERS_VARIED = (
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
        N_VARIED,
        VARIANCES,
        OUTLIERS_VARIED,
    )
    total = (
        len(DISTANCES)
        * len(MODELS)
        * ITERATIONS  # seeds
        * len(N_VARIED)
        * len(VARIANCES)
        * len(OUTLIERS_VARIED)
    )

    runner = Runner()
    experiment = Variances(cached=True)

    runner.run(experiment.get, iterator, unpack=True, length=total)


if __name__ == '__main__':
    main()
