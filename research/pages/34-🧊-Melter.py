#!/usr/bin/env python3

import io
import os.path

from matplotlib import pyplot as plt
import matplotlib.ticker as tck
import seaborn as sns
import streamlit as st

from viewer_extras.parameters import plot_options
from viewer_extras.parameters import st_page_options
from viewer_extras.utils import natural_sort
from viewer_extras.utils import load_dataframe


#
# Header and settings
#
st.set_page_config(**st_page_options)
st.title('Melter')

st.write('Present the real-world data results with automatic output grouping.')


#
# Select file
#
file = st.radio(
    label='File to view:',
    options=[
        '20newsgroups',
        'banking77',
        'cifar10',
        'cifar100',
        'ImageNet',
    ],
    index=4,
    horizontal=True,
)


#
# Data loading
#
try:
    df = load_dataframe(path=os.path.join('data',
                                          f'ood-per-class-{file}.pickle'))
except Exception as err:
    st.error(f'{type(err).__name__}: {err}')
    st.stop()


#
# Additional input
#
with st.sidebar:
    xkcd = st.checkbox('XKCD style', value=False)
    BERT_unify = st.checkbox('BERT unify', value=False)

    cols = st.columns(2)
    with cols[0]:
        xgrid = st.selectbox(
            label='X grid',
            options=(
                'both',
                'major',
                'minor',
                None,
            ),
        )
    with cols[1]:
        ygrid = st.selectbox(
            label='Y grid',
            options=(
                'both',
                'major',
                'minor',
                None,
            ),
        )


#
# Unify results for various BERT variants – their outcomes are very similar
#
# NOTE: Change in place causes some cache issues, so we create a new df here
#
if BERT_unify:
    BERT_hack = {
        'BERT-base-32': 'BERT-base',
        'BERT-base-r': 'BERT-base',
        'BERT-tiny-10': 'BERT-tiny',
        'BERT-tiny-64': 'BERT-tiny',
        'BERT-tiny-s': 'BERT-tiny',
        'BERT-tiny-s-64': 'BERT-tiny',
    }

    for to_replace, replacement in BERT_hack.items():
        df = df.replace(to_replace, replacement)


#
# Application input
#
with st.container(border=True):
    st.header('Settings', divider='rainbow')
    first_row = st.columns(2)
    second_row = st.columns(2)
    third_row = st.columns(2)
    filters = {}

    inputs = list(df.columns)[:4]
    outputs = list(df.columns)[5:]

    #
    # First row
    #
    with first_row[0]:
        x_axis = st.selectbox(
            label='X-axis (Groups)',
            options=inputs,
            index=0,
        )

    with first_row[1]:
        to_melt = st.multiselect(
            label='Y-axis (Output – columns to melt)',
            options=outputs,
            default=['sens_95', 'spec_95'],
        )

    #
    # Second row
    #
    with second_row[0]:
        key = 'representation'
        options = df[key].drop_duplicates().sort_values()
        filters[key] = st.multiselect(
            label=key.title(),
            options=options,
            default=options,
        )
        filters[key] = natural_sort(filters[key])

    with second_row[1]:
        key = 'class'
        options = df[key].drop_duplicates().sort_values()
        min_value = min(options)
        max_value = max(options)
        filters[key] = st.slider(
            label=key.title(),
            min_value=min_value,
            max_value=max_value,
            value=(min_value, max_value),
        )

    #
    # Third row
    #
    with third_row[0]:
        key = 'data'
        options = df[key].drop_duplicates().sort_values()
        filters[key] = st.multiselect(
            label=key.title(),
            options=options,
            default=options,
        )
        filters[key] = natural_sort(filters[key])

    with third_row[1]:
        key = 'model'
        options = df[key].drop_duplicates().sort_values()
        filters[key] = st.multiselect(
            label=key.title(),
            options=options,
            default=options,
        )
        filters[key] = natural_sort(filters[key])


#
# Filter rows
#
for parameter, values in filters.items():
    if parameter == 'class':
        df = df.loc[df[parameter].between(*values, inclusive='both')]
    else:
        df = df.loc[df[parameter].isin(values)]

df = df.melt(
    id_vars=filters.keys(),
    value_vars=to_melt,
    var_name='_melted_keys',
    value_name='_melted_values',
)


#
# Verify there is a result
#
if df.empty:
    st.error('No result for a given set of parameters!')
    st.stop()


#
# Generate filename
#
base_filename = f'{file}'
base_filename += f'-{",".join(to_melt)}({x_axis})'

for parameter, values in filters.items():
    base_filename += f'-{parameter}:'
    base_filename += ','.join(str(value) for value in values)

melter_filename = f'melter-{base_filename}.pdf'


#
# Content
#
left_column, right_column = st.columns(2)

if xkcd:
    plt.xkcd()
else:
    plt.rcdefaults()


#
# Customizing the plot
#
with right_column:
    with st.container(border=True):
        st.header('Customization', divider='rainbow')

        cols = st.columns([1, 2, 2])
        with cols[0]:
            st.markdown('# ')  # Spacing hack
            st.write('Labels:')
        with cols[1]:
            x_label = st.text_input(
                label='X label',
                value='',
                placeholder='(Leave empty for default)',
            )
        with cols[2]:
            y_label = st.text_input(
                label='Y label',
                value='',
                placeholder='(Leave empty for default)',
            )

        cols = st.columns([1, 2, 2])
        with cols[0]:
            st.markdown('# ')  # Spacing hack
            y_lim = st.checkbox('Y Limit', value=False)
        with cols[1]:
            y_min = st.number_input(
                label='Y Min',
                value=0.0,
                disabled=(not y_lim),
            )
        with cols[2]:
            y_max = st.number_input(
                label='Y Max',
                value=100.0,
                disabled=(not y_lim),
            )

        y_log = st.checkbox('Log scale on Y-axis', value=False)

        cols = st.columns([1, 2, 2])
        with cols[0]:
            st.markdown('# ')  # Spacing hack
            st.write('Fig size:')
        with cols[1]:
            fig_width = st.number_input(
                label='Width',
                value=9,
            )
        with cols[2]:
            fig_height = st.number_input(
                label='Height',
                value=5,
            )


#
# Drawing the plot
#
with left_column:
    with st.container(border=True):
        st.header('Plot', divider='rainbow')
        fig, ax = plt.subplots()

        sns.barplot(
            data=df,
            x=x_axis,
            y='_melted_values',
            hue='_melted_keys',
            order=filters[x_axis],
            estimator='median',
            errorbar=('pi', 95),
            linewidth=0.5,
            capsize=0.25,
            err_kws={'linewidth': 1.5},
        )

        ax.set_xlabel(x_label if x_label else x_axis)
        ax.set_ylabel(y_label if y_label else 'value')

        if y_lim:
            ax.set_ylim([y_min, y_max])

        if y_log:
            ax.set_yscale('log')
        else:
            ax.locator_params(nbins=10, axis='y')
            ax.yaxis.set_minor_locator(tck.AutoMinorLocator())

        if xgrid:
            ax.grid(which=xgrid, axis='x', linewidth=0.5, linestyle='dotted')
        if ygrid:
            ax.grid(which=ygrid, axis='y', linewidth=0.5, linestyle='dotted')

        ax.legend(
            loc='upper center',
            bbox_to_anchor=(0.5, 1.05),
            ncol=99,
            fancybox=True,
            shadow=True,
        )

        fig.set_figwidth(fig_width)
        fig.set_figheight(fig_height)
        fig.tight_layout()
        st.pyplot(fig, **plot_options)

        buffer = io.BytesIO()
        fig.savefig(buffer, format='pdf', **plot_options)

        st.download_button(
            label='Download PDF',
            data=buffer,
            file_name=melter_filename,
            mime='application/pdf',
        )
