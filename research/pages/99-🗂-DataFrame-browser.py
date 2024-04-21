#!/usr/bin/env python3

import os.path

import streamlit as st

from viewer_utils import load_dataframe


#
# Header and settings
#
st.set_page_config(layout='wide')
st.title('DataFrame browser')


#
# Select file
#
file = st.selectbox(
    label='File to view:',
    options=[
        'correlations.pickle',
        'distributions.pickle',
        'overlapping.pickle',
        'properties.pickle',
    ],
)


#
# Data loading
#
try:
    df = load_dataframe(path=os.path.join('data', file))
except Exception as err:
    st.error(f'{type(err).__name__}: {err}')
    st.stop()

df_columns = list(df.columns)[1:]  # we omit the rows IDs
df_rows = len(df)


#
# Application input
#
with st.container(border=True):
    st.header('Filters', divider='rainbow')

    query = st.text_input(
        label='Additional Pandas query:',
        value='',
        placeholder='E.g. dimension <= 1000 and samples == 500',
    )

    query_log = st.empty()

    columns_to_show = st.multiselect(
        label='Columns to show:',
        options=df_columns,
        default=df_columns,
    )


#
# Additional input
#
with st.sidebar:
    limit = st.number_input(
        label='Rows limit for preview',
        value=100,
    )


#
# Filter rows
#
if query:
    try:
        df = df.query(query.replace(' = ', ' == '))
    except Exception as err:
        query_log.error(f'{type(err).__name__}: {err}')
        df = df.iloc[0:0]  # Return no rows, but keep columns


#
# Filter columns
#
df = df[columns_to_show]


#
# Display dataframe
#
with st.container(border=True):
    st.header('DataFrame', divider='rainbow')
    st.write(f'{len(df)} rows filtered out of {df_rows}')
    st.dataframe(df[:limit])
