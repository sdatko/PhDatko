#!/usr/bin/env -S streamlit run

import streamlit as st

from viewer_extras.parameters import st_page_options

st.set_page_config(**st_page_options)
st.title('Hello World!')

st.info('Select a tool from the sidebar menu on the left.')

st.markdown('''
    - **Viewer** – leads to the main page and this description ;-)
    - **📊 Histograms** – allows to analyse the distributions of measures
                          and the separability of the data.
    - **📈 Trends** – makes it easier to determine the characteristics
                      and stability of measures as a function
                      of input parameters.
    - **🧮 Barplots** – displays the results of real-world data
                        with pre-defined groups.
    - **🧊 Melter** – presents the real-world data results
                      with automatic output grouping.
    - **🏡 Properties** – offers examination of the characteristics
                          of the data representations techniques.
    - **🖼️ Measures-illustrated** – contains figures depicting the outlierness
                                    measures and their properties.
    - **🗂 DataFrame-browser** – offers a filtered view of the data
                                 in the form of a table.
''')

st.markdown('# ')  # Spacing hack

with st.expander('Clearing the cache'):
    st.warning('If the file containing the data has changed after starting'
               ' the Streamlit application, please click on the button below.',
               icon="⚠️")

    if st.button('Clear All Cache'):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.success('Done – the cache is clear now!')
