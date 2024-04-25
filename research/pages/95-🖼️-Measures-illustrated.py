#!/usr/bin/env python3

import io

import streamlit as st

from viewer_extras.measures import abof_angles
from viewer_extras.measures import abof_distance
from viewer_extras.measures import euclidean_distance
from viewer_extras.measures import irwd_distance
from viewer_extras.measures import irwd_issue
from viewer_extras.measures import knn_distance
from viewer_extras.measures import lof_distance
from viewer_extras.measures import mahalanobis_distance
from viewer_extras.measures import seuclidean_distance
from viewer_extras.parameters import plot_options


#
# Header and settings
#
st.set_page_config(layout='wide')
st.title('Measures illustrated')

st.write('The figures below are involved in the Chapter 2 of Dissertation.')


#
# Helper function
#
def section(title, function, filename):
    with st.container(border=True):
        st.header(title, divider='rainbow')

        with st.spinner('Loading...'):
            fig = function()
            st.pyplot(fig, **plot_options)

            buffer = io.BytesIO()
            fig.savefig(buffer, format='pdf', **plot_options)

        st.download_button(
            label='Download PDF',
            data=buffer,
            file_name=filename,
            mime='application/pdf',
        )


#
# Content
#
column1, column2 = st.columns(2)

with column1:
    section('Angle-Based Outlier Factor – angles',
            abof_angles,
            'abof-angles.pdf')

    section('Angle-Based Outlier Factor',
            abof_distance,
            'abof-distance.pdf')

    section('Euclidean distance',
            euclidean_distance,
            'euclidean-distance.pdf')

    section('Integrated Rank Weighted Depth',
            irwd_distance,
            'irwd-distance.pdf')

    section('Integrated Rank Weighted Depth – issue',
            irwd_issue,
            'irwd-issue.pdf')

with column2:
    section('k-Nearest Neighbors',
            knn_distance,
            'knn-distance.pdf')

    section('Local Outlier Factor',
            lof_distance,
            'lof-distance.pdf')

    section('Mahalanobis distance',
            mahalanobis_distance,
            'mahalanobis-distance.pdf')

    section('Standardized Euclidean distance',
            seuclidean_distance,
            'seuclidean-distance.pdf')
