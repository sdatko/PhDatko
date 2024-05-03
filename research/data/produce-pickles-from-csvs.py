#!/usr/bin/env python3

import pickle
from time import asctime

import pandas as pd


#
# Helper functions
#
def rename_columns(df: pd.DataFrame) -> None:
    df.rename(
        columns={
            'model': 'representation',
            'OOD_method': 'model',
            '            skew': 'skew',
        },
        inplace=True,
    )


def reorder_columns(df: pd.DataFrame) -> None:
    column = df.pop('n')
    df.insert(4, column.name, column)


def replace_values(df: pd.DataFrame) -> None:
    values = {
        # models
        'Maha': 'MD',
        'Maha_pooled': 'MDP',
        'Minkowski': 'ED',
        'SEuclidean': 'SED',
        # representations
        'bert-base-32_0': 'BERT-base-32',
        'bert-base-range_0': 'BERT-base-r',
        'bert-tiny-10_0': 'BERT-tiny-10',
        'bert-tiny-64_0': 'BERT-tiny-64',
        'bert-tiny-s-64_0': 'BERT-tiny-s',
        'doc2vec_0': 'Doc2Vec',
        'fasttext_0': 'fastText',
        'tfidf_0': 'TF-IDF',
        # data
        'IDtest_other': 'other-test',
        'IDtrain_other': 'other-train',
        'imagenet_o': 'ImageNetO',
        'ood_cifar10': 'cifar10',
        'ood_cifar100': 'cifar100',
        'ood_svhn': 'SVHN',
        'OpenImage_O': 'OpenImageO',
        'test': 'ID-test',
        'train': 'ID-train',
    }

    for to_replace, replacement in values.items():
        df.replace(to_replace, replacement, inplace=True)


def process(filename: str, rename: bool, reorder: bool, replace: bool) -> None:
    print(asctime(), filename)
    df = pd.read_csv(filename)

    if rename:
        rename_columns(df)

    if reorder:
        reorder_columns(df)

    if replace:
        replace_values(df)

    with open(filename.replace('.csv', '.pickle'), 'wb') as file:
        pickle.dump(df, file, protocol=pickle.HIGHEST_PROTOCOL)


#
# Main
#
if __name__ == '__main__':
    settings = {
        'rename': True,
        'reorder': True,
        'replace': True,
    }
    process('ood-per-class-20newsgroups.csv', **settings)
    process('ood-per-class-banking77.csv', **settings)
    process('ood-per-class-cifar10.csv', **settings)
    process('ood-per-class-cifar100.csv', **settings)
    process('ood-per-class-ImageNet.csv', **settings)

    settings = {
        'rename': True,
        'reorder': False,
        'replace': True,
    }
    process('clusters-props-20newsgroups.csv', **settings)
    process('clusters-props-banking77.csv', **settings)
    process('clusters-props-cifar10.csv', **settings)
    process('clusters-props-cifar100.csv', **settings)
    process('clusters-props-ImageNet.csv', **settings)
