#!/usr/bin/env python3

import pickle

import streamlit as st


#
# Cached function for loading the data
#
# NOTE(sdatko): Paths in Streamlit are relative to the main file
#
# @st.cache_data  # Executes load to RAM on every rerun, slower after refresh
@st.cache_resource  # Shared between runs and sessions, instant after refresh
def load_dataframe(path='data/distributions.pickle'):
    with open(path, 'rb') as file:
        df = pickle.load(file)
    return df
