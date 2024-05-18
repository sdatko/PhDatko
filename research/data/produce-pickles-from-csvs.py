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
        'bert-base-32': 'BERT-base-32',  # clusters-props
        'bert-base-32_0': 'BERT-base-32',  # ood-per-class
        'bert-base-range': 'BERT-base-r',  # clusters-props
        'bert-base-range_0': 'BERT-base-r',  # ood-per-class
        'bert-tiny-10': 'BERT-tiny-10',  # clusters-props
        'bert-tiny-10_0': 'BERT-tiny-10',  # ood-per-class
        'bert-tiny-64': 'BERT-tiny-64',  # clusters-props
        'bert-tiny-64_0': 'BERT-tiny-64',  # ood-per-class
        'bert-tiny-s-64': 'BERT-tiny-s',  # clusters-props
        'bert-tiny-s-64_0': 'BERT-tiny-s',  # ood-per-class
        'doc2vec_0': 'Doc2Vec',  # both
        'fasttext': 'fastText',  # clusters-props
        'fasttext_0': 'fastText',  # ood-per-class
        'tfidf': 'TF-IDF',  # clusters-props
        'tfidf_0': 'TF-IDF',  # ood-per-class
        # data
        'IDtest_other': 'other-test',
        'IDtrain_other': 'other-train',
        'imagenet_o': 'ImageNet-O',
        'ood_cifar10': 'cifar10',
        'ood_cifar100': 'cifar100',
        'ood_svhn': 'SVHN',
        'OpenImage_O': 'OpenImage-O',
        'places365': 'Places365',
        'species': 'Species',
        'sun2012': 'SUN2012',
        'textures': 'Textures',
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
