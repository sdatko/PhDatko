#!/usr/bin/env python3

import io
import os.path

from matplotlib import pyplot as plt
import matplotlib.ticker as tck
import numpy as np
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
import streamlit as st

from viewer_extras.parameters import plot_options
from viewer_extras.parameters import st_page_options
from viewer_extras.utils import load_dataframe


#
# Header and settings
#
st.set_page_config(**st_page_options)
st.title('Histograms')

st.write('Analyse the distributions of measures'
         ' and the separability of the data.')


#
# Select file
#
file = st.radio(
    label='File to view:',
    options=[
        'correlations',
        'distributions',
        'variances',
    ],
    index=1,
    horizontal=True,
)


#
# Data loading
#
try:
    df = load_dataframe(path=os.path.join('data', f'{file}.pickle'))
except Exception as err:
    st.error(f'{type(err).__name__}: {err}')
    st.stop()


#
# Application input
#
with st.container(border=True):
    st.header('Settings', divider='rainbow')
    cols = st.columns(6)
    filters = {}

    if file == 'correlations':
        with cols[0]:
            key = 'n_correlated'
            filters[key] = st.select_slider(
                label=key.title(),
                options=df[key].drop_duplicates().sort_values(),
            )

        with cols[1]:
            key = 'covariance'
            filters[key] = st.select_slider(
                label=key.title(),
                options=df[key].drop_duplicates().sort_values(),
                value=1.00,
            )

        with cols[2]:
            key = 'distance'
            filters[key] = st.select_slider(
                label=key.title(),
                options=df[key].drop_duplicates().sort_values(),
            )

        with cols[3]:
            st.markdown('# ')  # Spacing hack
            key = 'outliers_correlated'
            filters[key] = st.checkbox(
                label=key.title(),
                value=False,
            )

        with cols[4]:
            key = 'model'
            filters[key] = st.selectbox(
                label=key.title(),
                options=df[key].drop_duplicates().sort_values(),
            )

        with cols[5]:
            key = 'seed'
            filters[key] = st.select_slider(
                label=key.title(),
                options=df[key].drop_duplicates().sort_values(),
            )

    if file == 'distributions':
        with cols[0]:
            key = 'dimension'
            filters[key] = st.select_slider(
                label=key.title(),
                options=df[key].drop_duplicates().sort_values(),
            )

        with cols[1]:
            key = 'samples'
            filters[key] = st.select_slider(
                label=key.title(),
                options=df[key].drop_duplicates().sort_values(),
            )

        with cols[2]:
            key = 'distance'
            filters[key] = st.select_slider(
                label=key.title(),
                options=df[key].drop_duplicates().sort_values(),
            )

        with cols[3]:
            key = 'distribution'
            filters[key] = st.selectbox(
                label=key.title(),
                options=df[key].drop_duplicates().sort_values(),
            )

        with cols[4]:
            key = 'model'
            filters[key] = st.selectbox(
                label=key.title(),
                options=df[key].drop_duplicates().sort_values(),
            )

        with cols[5]:
            key = 'seed'
            filters[key] = st.select_slider(
                label=key.title(),
                options=df[key].drop_duplicates().sort_values(),
            )

    if file == 'variances':
        with cols[0]:
            key = 'n_varied'
            filters[key] = st.select_slider(
                label=key.title(),
                options=df[key].drop_duplicates().sort_values(),
            )

        with cols[1]:
            key = 'variance'
            filters[key] = st.select_slider(
                label=key.title(),
                options=df[key].drop_duplicates().sort_values(),
            )

        with cols[2]:
            key = 'distance'
            filters[key] = st.select_slider(
                label=key.title(),
                options=df[key].drop_duplicates().sort_values(),
            )

        with cols[3]:
            st.markdown('# ')  # Spacing hack
            key = 'outliers_varied'
            filters[key] = st.checkbox(
                label=key.title(),
                value=False,
            )

        with cols[4]:
            key = 'model'
            filters[key] = st.selectbox(
                label=key.title(),
                options=df[key].drop_duplicates().sort_values(),
            )

        with cols[5]:
            key = 'seed'
            filters[key] = st.select_slider(
                label=key.title(),
                options=df[key].drop_duplicates().sort_values(),
            )

base_filename = f'{file}-{"-".join(f"{k}:{v}" for k, v in filters.items())}'
box_filename = f'box-{base_filename}.pdf'
hist_filename = f'hist-{base_filename}.pdf'
prc_filename = f'prc-{base_filename}.pdf'
roc_filename = f'roc-{base_filename}.pdf'


#
# Filter a dataframe for the single result
#
for key, wanted_value in filters.items():
    df = df.loc[df[key].values == wanted_value]


#
# Additional input
#
with st.sidebar:
    xkcd = st.checkbox('XKCD style', value=False)
    draw_optimal = st.checkbox('Draw optimal treshold', value=False)
    draw_tpr95 = st.checkbox('Draw TPR95 treshold', value=False)

    cols = st.columns(2)
    with cols[0]:
        fig_width = st.number_input(
            label='Figure Width',
            value=9,
        )
    with cols[1]:
        fig_height = st.number_input(
            label='Figure Height',
            value=5,
        )


#
# Verify there is a result
#
if df.empty:
    st.error('No result for a given set of parameters!')
    st.stop()


#
# Extract values from dataframe columns
#
# NOTE(sdatko): This is a bit silly, but apparently it must be done this way...
#               >>> df['train'].values
#               array([array([1.23, 4.56, 7.89, ...])], dtype=object)
#
train = np.array(df['train'].values[0])
known = np.array(df['known'].values[0])
unknown = np.array(df['unknown'].values[0])

if np.isnan(np.concatenate([train, known, unknown])).any():
    st.error('Invalid values in DataFrame!')
    st.dataframe(df)
    st.stop()


#
# Calculate the ROC
#
# NOTE: The roc_curve() assumes higher scores to be associated with
#       positive label (probability-like), however we assign greater
#       values with outliers, hence the order of 0-1 labels in y_true.
#
y_true = np.concatenate((np.zeros_like(known), np.ones_like(unknown)))
y_score = np.concatenate((known, unknown))

fpr, tpr, thresholds = roc_curve(y_true, y_score)
auroc = auc(fpr, tpr)

optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]

tpr95_idx = np.argmax(tpr >= 0.95)
tpr95_threshold = thresholds[tpr95_idx]

precision, recall, thresholds = precision_recall_curve(y_true, y_score)
aupr = auc(recall, precision)


#
# Content
#
left_column, right_column = st.columns(2)

if xkcd:
    plt.xkcd()
else:
    plt.rcdefaults()


#
# Plot the distributions
#
with left_column:
    with st.container(border=True):
        st.header('Distributions', divider='rainbow')
        fig, ax = plt.subplots()

        all_data = np.concatenate([known, unknown, train])
        data_range = (all_data.min(), all_data.max())
        hist_options = {
            'range': data_range,
            'bins': 30,
        }

        ax.hist(known, alpha=0.5, label='known', **hist_options)
        ax.hist(unknown, alpha=0.5, label='unknown', **hist_options)
        ax.hist(train, alpha=0.5, label='train', **hist_options)

        if draw_optimal:
            ax.axvline(
                x=optimal_threshold,
                color='red',
                linestyle='--',
                label='optimal threshold',
            )

        if draw_tpr95:
            ax.axvline(
                x=tpr95_threshold,
                color='green',
                linestyle='-.',
                label='TPR95 threshold',
            )

        ax.set_xlabel('Outlierness score')
        ax.set_ylabel('Frequency')

        ax.locator_params(nbins=10, axis='both')
        ax.xaxis.set_minor_locator(tck.AutoMinorLocator())
        ax.yaxis.set_minor_locator(tck.AutoMinorLocator())

        ax.grid(which='both', axis='both', linewidth=0.5, linestyle='dotted')
        ax.legend(loc='upper right')

        fig.set_figwidth(fig_width)
        fig.set_figheight(fig_height)
        fig.tight_layout()
        st.pyplot(fig, **plot_options)

        buffer = io.BytesIO()
        fig.savefig(buffer, format='pdf', **plot_options)

        st.download_button(
            label='Download PDF',
            data=buffer,
            file_name=hist_filename,
            mime='application/pdf',
        )

    with st.container(border=True):
        st.header('BoxPlots', divider='rainbow')
        fig, ax = plt.subplots()

        all_data = [train, known, unknown]

        ax.boxplot(all_data, vert=False, showcaps=False, widths=0.75)
        ax.violinplot(all_data, vert=False, widths=0.5)

        if draw_optimal:
            ax.axvline(
                x=optimal_threshold,
                color='red',
                linestyle='--',
                label='optimal threshold',
            )

        if draw_tpr95:
            ax.axvline(
                x=tpr95_threshold,
                color='green',
                linestyle='-.',
                label='TPR95 threshold',
            )

        ax.set_xlabel('Outlierness score')
        ax.set_yticks([1, 2, 3], labels=['train', 'known', 'unknown'])

        ax.locator_params(nbins=10, axis='both')
        ax.xaxis.set_minor_locator(tck.AutoMinorLocator())

        ax.grid(which='both', axis='both', linewidth=0.5, linestyle='dotted')

        fig.set_figwidth(fig_width)
        fig.set_figheight(fig_height)
        fig.tight_layout()
        st.pyplot(fig, **plot_options)

        buffer = io.BytesIO()
        fig.savefig(buffer, format='pdf', **plot_options)

        st.download_button(
            label='Download PDF',
            data=buffer,
            file_name=box_filename,
            mime='application/pdf',
        )


#
# Plot the ROC
#
with right_column:
    with st.container(border=True):
        st.header('Receiver operating characteristic', divider='rainbow')
        fig, ax = plt.subplots()

        ax.plot(fpr, tpr, color='darkorange', linewidth=1,
                label=f'ROC (area = {auroc:0.3f})' % auroc)
        ax.plot([0, 1], [0, 1], color='navy', linewidth=1, linestyle='--')
        ax.fill_between(fpr, tpr, 0, facecolor='darkorange', alpha=0.1)

        if draw_optimal:
            ax.plot(
                fpr[optimal_idx],
                tpr[optimal_idx],
                marker='o',
                markersize=10,
                color='red',
                alpha=0.5,
            )

        if draw_tpr95:
            ax.plot(
                fpr[tpr95_idx],
                tpr[tpr95_idx],
                marker='o',
                markersize=10,
                color='green',
                alpha=0.5,
            )

        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.0])

        ax.locator_params(nbins=10, axis='both')
        ax.xaxis.set_minor_locator(tck.AutoMinorLocator())
        ax.yaxis.set_minor_locator(tck.AutoMinorLocator())

        ax.grid(which='both', axis='both', linewidth=0.5, linestyle='dotted')
        ax.legend(loc='lower right')

        fig.set_figwidth(fig_width)
        fig.set_figheight(fig_height)
        fig.tight_layout()
        st.pyplot(fig, **plot_options)

        buffer = io.BytesIO()
        fig.savefig(buffer, format='pdf', **plot_options)

        st.download_button(
            label='Download PDF',
            data=buffer,
            file_name=roc_filename,
            mime='application/pdf',
        )

    with st.container(border=True):
        st.header('Precision–Recall Curve', divider='rainbow')
        fig, ax = plt.subplots()

        ax.plot(recall, precision, color='darkorange', linewidth=1,
                label=f'PR (area = {aupr:0.3f})' % auroc)
        ax.fill_between(recall, precision, 0, facecolor='darkorange', alpha=.1)

        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.0])

        ax.locator_params(nbins=10, axis='both')
        ax.xaxis.set_minor_locator(tck.AutoMinorLocator())
        ax.yaxis.set_minor_locator(tck.AutoMinorLocator())

        ax.grid(which='both', axis='both', linewidth=0.5, linestyle='dotted')
        ax.legend(loc='lower right')

        fig.set_figwidth(fig_width)
        fig.set_figheight(fig_height)
        fig.tight_layout()
        st.pyplot(fig, **plot_options)

        buffer = io.BytesIO()
        fig.savefig(buffer, format='pdf', **plot_options)

        st.download_button(
            label='Download PDF',
            data=buffer,
            file_name=prc_filename,
            mime='application/pdf',
        )
