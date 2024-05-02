#!/usr/bin/env python3

import pickle
from time import asctime

import numpy as np
import pandas as pd
from pony import orm
from sklearn.metrics import auc
from sklearn.metrics import roc_curve

from openset.experiments.correlations import Correlations
from openset.experiments.distributions import Generated
from openset.experiments.overlapping import BoundingBoxes
from openset.experiments.properties import MVNEstimation
from openset.experiments.variances import Variances


#
# Helper functions
#
def quartiles(df: pd.DataFrame) -> None:
    '''Calculate the distributions statistics and save results in-place.

    Computes the five number summary for each dataset and IQR distances.
    '''
    df['train_q0'] = [row[0] for row in df['train'].values]
    df['train_q1'] = [row[25] for row in df['train'].values]
    df['train_q2'] = [row[50] for row in df['train'].values]
    df['train_q3'] = [row[75] for row in df['train'].values]
    df['train_q4'] = [row[100] for row in df['train'].values]

    df['known_q0'] = [row[0] for row in df['known'].values]
    df['known_q1'] = [row[25] for row in df['known'].values]
    df['known_q2'] = [row[50] for row in df['known'].values]
    df['known_q3'] = [row[75] for row in df['known'].values]
    df['known_q4'] = [row[100] for row in df['known'].values]

    df['unknown_q0'] = [row[0] for row in df['unknown'].values]
    df['unknown_q1'] = [row[25] for row in df['unknown'].values]
    df['unknown_q2'] = [row[50] for row in df['unknown'].values]
    df['unknown_q3'] = [row[75] for row in df['unknown'].values]
    df['unknown_q4'] = [row[100] for row in df['unknown'].values]

    df['train_iqr'] = df['train_q3'].values - df['train_q1'].values
    df['known_iqr'] = df['known_q3'].values - df['known_q1'].values
    df['unknown_iqr'] = df['unknown_q3'].values - df['unknown_q1'].values

    df['known_iqr_dist'] = (
        abs(df['known_q2'].values - df['train_q2'].values)
        / df['train_iqr'].values
    )
    df['unknown_iqr_dist'] = (
        abs(df['unknown_q2'].values - df['train_q2'].values)
        / df['train_iqr'].values
    )


def classification(df: pd.DataFrame, treshold: int = 95) -> None:
    '''Perform train-wise classification and save results in-place.

    Hypothesis: data is outlier;
    - TN – known labeled as inlier,
    - FP – known labeled as outlier,
    - TP – unknown labeled as outlier,
    - FN – unknown labeled as inlier.
    '''
    df[f'TN_{treshold}'] = [
        sum(known <= train[treshold])
        for train, known in zip(df['train'].values, df['known'].values)
    ]
    df[f'FP_{treshold}'] = [
        sum(known > train[treshold])
        for train, known in zip(df['train'].values, df['known'].values)
    ]
    df[f'TP_{treshold}'] = [
        sum(unknown > train[treshold])
        for train, unknown in zip(df['train'].values, df['unknown'].values)
    ]
    df[f'FN_{treshold}'] = [
        sum(unknown <= train[treshold])
        for train, unknown in zip(df['train'].values, df['unknown'].values)
    ]

    df[f'precision_{treshold}'] = (
        df[f'TP_{treshold}'].values
        /
        (
            df[f'TP_{treshold}'].values
            + df[f'FP_{treshold}'].values
        )
    )
    df[f'recall_{treshold}'] = (
        df[f'TP_{treshold}'].values
        /
        (
            df[f'TP_{treshold}'].values
            + df[f'FN_{treshold}'].values
        )
    )

    df[f'spec_{treshold}'] = (
        df[f'TN_{treshold}'].values
        /
        (
            df[f'FP_{treshold}'].values
            + df[f'TN_{treshold}'].values
        )
    )
    df[f'sens_{treshold}'] = df[f'recall_{treshold}'].values

    df[f'F1_{treshold}'] = (
        (
            2
            * df[f'precision_{treshold}'].values
            * df[f'recall_{treshold}'].values
        )
        /
        (
            df[f'precision_{treshold}'].values
            + df[f'recall_{treshold}'].values
        )
    )

    df[f'accuracy_{treshold}'] = (
        (
            df[f'TP_{treshold}'].values
            + df[f'TN_{treshold}'].values
        )
        /
        (
            df[f'TP_{treshold}'].values
            + df[f'FP_{treshold}'].values
            + df[f'TN_{treshold}'].values
            + df[f'FN_{treshold}'].values
        )
    )


def separability(df: pd.DataFrame) -> None:
    '''Determine how well the data can be separated and save results in-place.

    Calculate the Area Under the Receiver Operating Characteristic curve.
    '''
    def auroc(inliers, outliers) -> float:
        if any(np.isnan(inliers)) or any(np.isnan(outliers)):
            return None

        # NOTE: The roc_curve() assumes higher scores to be associated with
        #       positive label (probability-like), however we assign greater
        #       values with outliers, hence the order of 0-1 labels in y_true.
        y_true = np.concatenate((np.zeros_like(inliers),
                                 np.ones_like(outliers)))
        y_score = np.concatenate((inliers, outliers))

        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        return auc(fpr, tpr)

    df['auroc'] = [auroc(row.known, row.unknown)
                   for row in df.itertuples(index=False)]


def intersection_over_union(df: pd.DataFrame) -> None:
    '''Calculates the Jaccard index of bounding boxes.

    It is defined as the ratio between the overlapping area/volume
    and the union of the areas/volumes.
    J(A, B) = |A∩B| / |A∪B| = |A∩B| / (|A| + |B| - |A∩B|)
    '''
    volume1 = 100 * df['volume'] / df['factor1']
    volume2 = 100 * df['volume'] / df['factor2']

    df['IoU'] = df['volume'] / (volume1 + volume2 - df['volume'])


#
# Distributions of measures experiment
#
def distributions():
    Generated.setup_db()

    with orm.db_session():
        print(asctime(), 'Selecting...')
        rows = Generated.Cache.select()
        entries = []

        print(asctime(), 'Generating DataFrame...')
        #
        # NOTE(sdatko): By default the array data in Pony ORM are kept under
        #               the pony.orm.ormtypes.TrackedArray type, so we need
        #               to convert all such values individually in a loop
        #               instead of just using pd.DataFrame(row.to_dict()...)
        #
        # df = pd.DataFrame(row.to_dict() for row in rows)  # see note
        for row in rows:
            entry = row.to_dict()

            entry['train'] = np.array(entry['train'].get_untracked())
            entry['known'] = np.array(entry['known'].get_untracked())
            entry['unknown'] = np.array(entry['unknown'].get_untracked())

            entries.append(entry)

        df = pd.DataFrame(entries)

    print(asctime(), 'Processing additional columns...')
    df = df.drop('id', axis=1)  # Omit the rows IDs (SQL primary key)
    classification(df, 90)
    classification(df, 95)
    classification(df, 99)
    quartiles(df)
    separability(df)

    print(asctime(), 'Saving...')
    with open('distributions.pickle', 'wb') as file:
        pickle.dump(df, file, protocol=pickle.HIGHEST_PROTOCOL)

    print(asctime(), 'Done')


#
# Correlations experiment
#
def correlations():
    Correlations.setup_db()

    with orm.db_session():
        print(asctime(), 'Selecting...')
        rows = Correlations.Cache.select()
        entries = []

        print(asctime(), 'Generating DataFrame...')
        #
        # NOTE(sdatko): By default the array data in Pony ORM are kept under
        #               the pony.orm.ormtypes.TrackedArray type, so we need
        #               to convert all such values individually in a loop
        #               instead of just using pd.DataFrame(row.to_dict()...)
        #
        # df = pd.DataFrame(row.to_dict() for row in rows)  # see note
        for row in rows:
            entry = row.to_dict()

            entry['train'] = np.array(entry['train'].get_untracked())
            entry['known'] = np.array(entry['known'].get_untracked())
            entry['unknown'] = np.array(entry['unknown'].get_untracked())

            entries.append(entry)

        df = pd.DataFrame(entries)

    print(asctime(), 'Processing additional columns...')
    df = df.drop('id', axis=1)  # Omit the rows IDs (SQL primary key)
    classification(df, 90)
    classification(df, 95)
    classification(df, 99)
    quartiles(df)
    separability(df)

    print(asctime(), 'Saving...')
    with open('correlations.pickle', 'wb') as file:
        pickle.dump(df, file, protocol=pickle.HIGHEST_PROTOCOL)

    print(asctime(), 'Done')


#
# Variances experiment
#
def variances():
    Variances.setup_db()

    with orm.db_session():
        print(asctime(), 'Selecting...')
        rows = Variances.Cache.select()
        entries = []

        print(asctime(), 'Generating DataFrame...')
        #
        # NOTE(sdatko): By default the array data in Pony ORM are kept under
        #               the pony.orm.ormtypes.TrackedArray type, so we need
        #               to convert all such values individually in a loop
        #               instead of just using pd.DataFrame(row.to_dict()...)
        #
        # df = pd.DataFrame(row.to_dict() for row in rows)  # see note
        for row in rows:
            entry = row.to_dict()

            entry['train'] = np.array(entry['train'].get_untracked())
            entry['known'] = np.array(entry['known'].get_untracked())
            entry['unknown'] = np.array(entry['unknown'].get_untracked())

            entries.append(entry)

        df = pd.DataFrame(entries)

    print(asctime(), 'Processing additional columns...')
    df = df.drop('id', axis=1)  # Omit the rows IDs (SQL primary key)
    classification(df, 90)
    classification(df, 95)
    classification(df, 99)
    quartiles(df)
    separability(df)

    print(asctime(), 'Saving...')
    with open('variances.pickle', 'wb') as file:
        pickle.dump(df, file, protocol=pickle.HIGHEST_PROTOCOL)

    print(asctime(), 'Done')


#
# Bounding boxes overlapping experiment
#
def overlapping():
    BoundingBoxes.setup_db()

    with orm.db_session():
        print(asctime(), 'Selecting...')
        rows = BoundingBoxes.Cache.select()

        print(asctime(), 'Generating DataFrame...')
        df = pd.DataFrame(row.to_dict() for row in rows)

    print(asctime(), 'Processing additional columns...')
    df = df.drop('id', axis=1)  # Omit the rows IDs (SQL primary key)
    intersection_over_union(df)

    print(asctime(), 'Saving...')
    with open('overlapping.pickle', 'wb') as file:
        pickle.dump(df, file, protocol=pickle.HIGHEST_PROTOCOL)

    print(asctime(), 'Done')


#
# MVN properties estimation experiment
#
def properties():
    MVNEstimation.setup_db()

    with orm.db_session():
        print(asctime(), 'Selecting...')
        rows = MVNEstimation.Cache.select()

        print(asctime(), 'Generating DataFrame...')
        df = pd.DataFrame(row.to_dict() for row in rows)

    print(asctime(), 'Processing additional columns...')
    df = df.drop('id', axis=1)  # Omit the rows IDs (SQL primary key)

    print(asctime(), 'Saving...')
    with open('properties.pickle', 'wb') as file:
        pickle.dump(df, file, protocol=pickle.HIGHEST_PROTOCOL)

    print(asctime(), 'Done')


#
# Main
#
if __name__ == '__main__':
    distributions()
    correlations()
    variances()
    overlapping()
    properties()
