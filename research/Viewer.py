#!/usr/bin/env -S streamlit run

import streamlit as st

st.set_page_config(
    initial_sidebar_state='expanded',
    layout='wide',
)

st.title('Hello World!')

with st.expander('Clearing the cache'):
    st.warning('If the file containing the data has changed after starting'
               ' the Streamlit application, please click on the button below.',
               icon="⚠️")

    if st.button('Clear All Cache'):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.success('Done – the cache is clear now!')
