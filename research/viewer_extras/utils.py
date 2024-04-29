#!/usr/bin/env python3

from datetime import datetime
import pickle
import re

import streamlit as st


#
# Natural sort for a list of strings
#
# Code adapted from: https://stackoverflow.com/a/4836734
#
def natural_sort(values):
    def alphanum_key(value):
        return [int(part) if part.isdigit() else part.lower()
                for part in re.split('([0-9]+)', str(value))]

    return sorted(values, key=alphanum_key)


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


#
# Returns current date and time with milliseconds
#
# E.g. 2023-03-14-21-37-54-789456
#
# def strfnow(fmt='%Y-%m-%d-%H-%M-%S-%f'):
def now(fmt='%Y-%m-%d-%H-%M-%S-%f'):
    return datetime.now().strftime(fmt)
