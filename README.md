PhDatko
=======

This repository contains source files of the tooling I used
during my PhD research and work on the final dissertation.


Viewer application
------------------

To run the Viewer application, install the dependencies specified in the
`requirements.txt` file, then run the `Viewer.py` from `research/` directory.

Alternatively, this can be achieved conveniently with the `tox` command:

```
tox -e streamlit
```


Data for the Viewer
-------------------

The Viewer application relies on the results of the scripts that are located
in the `research/data/` subdirectory. However, it may take up to several days
to generate everything from scratch. Hence, for convenience, the pre-computed
data for the Viewer application can be downloaded using the following command:

```
tox -e data
```

It runs the `fetch-precomputed-data.py` script from `research/data/` directory.
This needs to be performed usually only once.
