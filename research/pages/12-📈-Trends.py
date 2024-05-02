#!/usr/bin/env python3

import io
import os.path

from matplotlib import pyplot as plt
import matplotlib.ticker as tck
import streamlit as st

from viewer_extras.parameters import legend_locations
from viewer_extras.parameters import plot_options
from viewer_extras.parameters import st_page_options
from viewer_extras.utils import natural_sort
from viewer_extras.utils import load_dataframe


#
# Header and settings
#
st.set_page_config(**st_page_options)
st.title('Trends')

st.write('Determine the characteristics and stability of parameters'
         ' as a function of selected input.')


#
# Select file
#
file = st.radio(
    label='File to view:',
    options=[
        'correlations.pickle',
        'distributions.pickle',
        'overlapping.pickle',
        'properties.pickle',
    ],
    index=1,
    horizontal=True,
)


#
# Data loading
#
try:
    df = load_dataframe(path=os.path.join('data', file))
except Exception as err:
    st.error(f'{type(err).__name__}: {err}')
    st.stop()


#
# Additional input
#
with st.sidebar:
    xkcd = st.checkbox('XKCD style', value=False)
    aggregate_seeds = st.checkbox(label='Aggregate seeds', value=False)


#
# Application input
#
with st.container(border=True):
    st.header('Settings', divider='rainbow')
    first_row = st.columns(2)
    second_row = st.columns([1, 2, 3])
    third_row = st.columns(6)
    filters = {}

    if file == 'correlations.pickle':
        x_inputs = [
            'n_correlated',  # 3
            'covariance',  # 4
            'distance',  # 0
            # 'model',  # 1
            # 'seed',  # 2
            # 'outliers_correlated',  # 5
        ]
        inputs = list(df.columns)[:6]
        outputs = list(df.columns)[9:]

    if file == 'distributions.pickle':
        x_inputs = [
            'dimension',  # 0
            'samples',  # 4
            'distance',  # 1
            # 'distribution',  # 2
            # 'model',  # 3
            # 'seed',  # 5
        ]
        inputs = list(df.columns)[:6]
        outputs = list(df.columns)[9:]

    if file == 'overlapping.pickle':
        x_inputs = [
            'dimension',  # 0
            'samples',  # 2
            # 'distribution',  # 1
            # 'seed',  # 3
        ]
        inputs = list(df.columns)[:4]
        outputs = list(df.columns)[4:]

    if file == 'properties.pickle':
        x_inputs = [
            'dimension',  # 0
            'samples',  # 1
            'n_correlated',  # 2
            'covariance',  # 3
            # 'seed',  # 4
        ]
        inputs = list(df.columns)[:5]
        outputs = list(df.columns)[5:]

    #
    # First row
    #
    with first_row[0]:
        x_axis = st.selectbox(
            label='X-axis (Input)',
            options=x_inputs,
        )

    with first_row[1]:
        y_axis = st.selectbox(
            label='Y-axis (Output)',
            options=outputs,
            index=(len(outputs) - 1),
        )

    #
    # Second row
    #
    with second_row[0]:
        st.markdown('# ')  # Spacing hack
        multi_plots = st.checkbox(
            label='Multiple plots',
            value=False,
        )

    with second_row[1]:
        parameter = st.selectbox(
            label='Parameter (used as label and for multi-plot)',
            options=[x for x in inputs if x != x_axis],
        )

    with second_row[2]:
        values = st.multiselect(
            label='Values for multi-plot',
            options=df[parameter].drop_duplicates().sort_values(),
            disabled=(not multi_plots),
        )
        values = natural_sort(values)

    #
    # Third row
    #
    if file == 'correlations.pickle':
        with third_row[0]:
            key = 'n_correlated'
            filters[key] = st.select_slider(
                label=key.title(),
                options=df[key].drop_duplicates().sort_values(),
                disabled=(x_axis == key or multi_plots and parameter == key),
            )

        with third_row[1]:
            key = 'covariance'
            filters[key] = st.select_slider(
                label=key.title(),
                options=df[key].drop_duplicates().sort_values(),
                value=1.00,
                disabled=(x_axis == key or multi_plots and parameter == key),
            )

        with third_row[2]:
            key = 'distance'
            filters[key] = st.select_slider(
                label=key.title(),
                options=df[key].drop_duplicates().sort_values(),
                disabled=(x_axis == key or multi_plots and parameter == key),
            )

        with third_row[3]:
            st.markdown('# ')  # Spacing hack
            key = 'outliers_correlated'
            filters[key] = st.checkbox(
                label=key.title(),
                value=False,
                disabled=(multi_plots and parameter == key),
            )

        with third_row[4]:
            key = 'model'
            filters[key] = st.selectbox(
                label=key.title(),
                options=df[key].drop_duplicates().sort_values(),
                disabled=(multi_plots and parameter == key),
            )

        with third_row[5]:
            key = 'seed'
            filters[key] = st.select_slider(
                label=key.title(),
                options=df[key].drop_duplicates().sort_values(),
                disabled=(aggregate_seeds or multi_plots and parameter == key),
            )

    if file == 'distributions.pickle':
        with third_row[0]:
            key = 'dimension'
            filters[key] = st.select_slider(
                label=key.title(),
                options=df[key].drop_duplicates().sort_values(),
                disabled=(x_axis == key or multi_plots and parameter == key),
            )

        with third_row[1]:
            key = 'samples'
            filters[key] = st.select_slider(
                label=key.title(),
                options=df[key].drop_duplicates().sort_values(),
                disabled=(x_axis == key or multi_plots and parameter == key),
            )

        with third_row[2]:
            key = 'distance'
            filters[key] = st.select_slider(
                label=key.title(),
                options=df[key].drop_duplicates().sort_values(),
                disabled=(x_axis == key or multi_plots and parameter == key),
            )

        with third_row[3]:
            key = 'distribution'
            filters[key] = st.selectbox(
                label=key.title(),
                options=df[key].drop_duplicates().sort_values(),
                disabled=(multi_plots and parameter == key),
            )

        with third_row[4]:
            key = 'model'
            filters[key] = st.selectbox(
                label=key.title(),
                options=df[key].drop_duplicates().sort_values(),
                disabled=(multi_plots and parameter == key),
            )

        with third_row[5]:
            key = 'seed'
            filters[key] = st.select_slider(
                label=key.title(),
                options=df[key].drop_duplicates().sort_values(),
                disabled=(aggregate_seeds or multi_plots and parameter == key),
            )

    if file == 'overlapping.pickle':
        with third_row[0]:
            key = 'dimension'
            filters[key] = st.select_slider(
                label=key.title(),
                options=df[key].drop_duplicates().sort_values(),
                disabled=(x_axis == key or multi_plots and parameter == key),
            )

        with third_row[1]:
            key = 'samples'
            filters[key] = st.select_slider(
                label=key.title(),
                options=df[key].drop_duplicates().sort_values(),
                disabled=(x_axis == key or multi_plots and parameter == key),
            )

        with third_row[2]:
            key = 'distribution'
            filters[key] = st.selectbox(
                label=key.title(),
                options=df[key].drop_duplicates().sort_values(),
                disabled=(multi_plots and parameter == key),
            )

        with third_row[3]:
            pass

        with third_row[4]:
            pass

        with third_row[5]:
            key = 'seed'
            filters[key] = st.select_slider(
                label=key.title(),
                options=df[key].drop_duplicates().sort_values(),
                disabled=(aggregate_seeds or multi_plots and parameter == key),
            )

    if file == 'properties.pickle':
        with third_row[0]:
            key = 'dimension'
            filters[key] = st.select_slider(
                label=key.title(),
                options=df[key].drop_duplicates().sort_values(),
                disabled=(x_axis == key or multi_plots and parameter == key),
            )

        with third_row[1]:
            key = 'samples'
            filters[key] = st.select_slider(
                label=key.title(),
                options=df[key].drop_duplicates().sort_values(),
                disabled=(x_axis == key or multi_plots and parameter == key),
            )

        with third_row[2]:
            key = 'n_correlated'
            filters[key] = st.select_slider(
                label=key.title(),
                options=df[key].drop_duplicates().sort_values(),
                disabled=(x_axis == key or multi_plots and parameter == key),
            )

        with third_row[3]:
            key = 'covariance'
            filters[key] = st.select_slider(
                label=key.title(),
                options=df[key].drop_duplicates().sort_values(),
                value=1.00,
                disabled=(x_axis == key or multi_plots and parameter == key),
            )

        with third_row[4]:
            pass

        with third_row[5]:
            key = 'seed'
            filters[key] = st.select_slider(
                label=key.title(),
                options=df[key].drop_duplicates().sort_values(),
                disabled=(aggregate_seeds or multi_plots and parameter == key),
            )

    #
    # Last row
    #
    query = st.text_input(
        label='Additional Pandas query:',
        value='',
        placeholder='E.g. dimension <= 1000',
    )

    query_log = st.empty()


#
# Filter rows
#
if aggregate_seeds:
    del filters['seed']

if x_axis in filters:
    del filters[x_axis]

if multi_plots:
    if parameter in filters:
        del filters[parameter]
    df = df.loc[df[parameter].isin(values)]

for key, wanted_value in filters.items():
    df = df.loc[df[key].values == wanted_value]

if query:
    try:
        df = df.query(query.replace(' = ', ' == '))
    except Exception as err:
        query_log.error(f'{type(err).__name__}: {err}')
        st.stop()


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
base_filename += f'-{y_axis}({x_axis})-'
base_filename += '-'.join(f'{k}:{v}' for k, v in filters.items())

if multi_plots:
    base_filename += f'-{parameter}:'
    base_filename += ','.join(str(value) for value in values)

if aggregate_seeds:
    base_filename += '-aggregated'

trend_filename = f'trend-{base_filename}.pdf'


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
            st.write('Legend:')
        with cols[1]:
            legend_location = st.selectbox(
                label='Legend location',
                options=legend_locations,
                index=1,
            )
        with cols[2]:
            legend_title = st.text_input(
                label='Legend title',
                value='',
                placeholder='(Leave empty to omit)',
            )

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
            x_lim = st.checkbox('X Limit', value=False)
        with cols[1]:
            x_min = st.number_input(
                label='X Min',
                value=0.0,
                disabled=(not x_lim),
            )
        with cols[2]:
            x_max = st.number_input(
                label='X Max',
                value=100.0,
                disabled=(not x_lim),
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

        cols = st.columns([1, 2, 2])
        with cols[0]:
            st.write('Log scale:')
        with cols[1]:
            x_log = st.checkbox('Log scale on X-axis', value=False)
        with cols[2]:
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

        # Details: https://matplotlib.org/stable/api/markers_api.html
        markers = ['o', '>', 's', '*', 'P', 'X', 'D', 'p']

        # Single plot = multi-plot with only one value
        if not multi_plots:
            values = [filters[parameter]]

        for value in values:
            df_single = df.loc[df[parameter].values == value].copy()
            df_single.sort_values(by=x_axis, inplace=True)

            if aggregate_seeds:
                df_single_agg = df_single.groupby([x_axis])
                x_values = df_single_agg[x_axis].mean()
                y_values = df_single_agg[y_axis].mean()
                yerr_values = df_single_agg[y_axis].std()
            else:
                x_values = df_single[x_axis]
                y_values = df_single[y_axis]
                yerr_values = None

            ax.errorbar(
                x_values,
                y_values,
                yerr=yerr_values,
                capsize=3,
                linewidth=1,
                linestyle='--',
                marker=markers[0],
                markersize=5,
                label=value,
                alpha=0.75,
            )
            markers = markers[1:] + markers[:1]  # rotate list of markers

        ax.set_xlabel(x_label if x_label else x_axis)
        ax.set_ylabel(y_label if y_label else y_axis)

        if x_lim:
            ax.set_xlim([x_min, x_max])
        if y_lim:
            ax.set_ylim([y_min, y_max])

        if x_log:
            ax.set_xscale('log')
        else:
            ax.locator_params(nbins=10, axis='x')
            ax.xaxis.set_minor_locator(tck.AutoMinorLocator())

        if y_log:
            ax.set_yscale('log')
        else:
            ax.locator_params(nbins=10, axis='y')
            ax.yaxis.set_minor_locator(tck.AutoMinorLocator())

        ax.grid(which='both', axis='both', linewidth=0.5, linestyle='dotted')
        legend = ax.legend(loc=legend_location)

        if legend_title:
            legend.set_title(legend_title)

        fig.set_figwidth(fig_width)
        fig.set_figheight(fig_height)
        fig.tight_layout()
        st.pyplot(fig, **plot_options)

        buffer = io.BytesIO()
        fig.savefig(buffer, format='pdf', **plot_options)

        st.download_button(
            label='Download PDF',
            data=buffer,
            file_name=trend_filename,
            mime='application/pdf',
        )
