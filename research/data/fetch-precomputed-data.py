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
