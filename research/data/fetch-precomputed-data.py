#!/usr/bin/env python3

from os.path import exists
from urllib.request import urlretrieve

REMOTE_BASE_URL = 'https://datko.pl/PhDatko/data/'
FILES_TO_GET = (
    'correlations.pickle',
    'correlations.sqlite',
    'distributions.pickle',
    'distributions.sqlite',
    'overlapping.pickle',
    'overlapping.sqlite',
    'properties.pickle',
    'properties.sqlite',
    'variances.pickle',
    'variances.sqlite',
    'ood-per-class-20newsgroups.csv',
    'ood-per-class-20newsgroups.pickle',
    'ood-per-class-banking77.csv',
    'ood-per-class-banking77.pickle',
    'ood-per-class-cifar10.csv',
    'ood-per-class-cifar10.pickle',
    'ood-per-class-cifar100.csv',
    'ood-per-class-cifar100.pickle',
    'ood-per-class-ImageNet.csv',
    'ood-per-class-ImageNet.pickle',
    'clusters-props-20newsgroups.csv',
    'clusters-props-20newsgroups.pickle',
    'clusters-props-banking77.csv',
    'clusters-props-banking77.pickle',
    'clusters-props-cifar10.csv',
    'clusters-props-cifar10.pickle',
    'clusters-props-cifar100.csv',
    'clusters-props-cifar100.pickle',
    'clusters-props-ImageNet.csv',
    'clusters-props-ImageNet.pickle',
)

print('Fetching files:')
for file in FILES_TO_GET:
    print(f'-> {file}', end=' ')
    if exists(file):
        print('[exists]')
    else:
        urlretrieve(f'{REMOTE_BASE_URL}/{file}', f'{file}')
        print('[ok]')

print('Done!')
