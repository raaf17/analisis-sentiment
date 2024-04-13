import streamlit as st
import numpy as np
import pandas as pd
from sklearn import datasets

def app():
    st.sidebar.subheader('Simpan Data Labeling')
    st.sidebar.text_input('Nama File Labeling :')
    st.sidebar.button('Simpan')
    
    st.header('Preprocessing Data')

    file_uploader = st.file_uploader('Upload Data Setelah Preprocessing :', ['csv'])

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ###### Original Text
        """)
        if file_uploader is not None:
            st.dataframe(file_uploader)
    with col2:
        st.markdown("""
        ###### Processed Text
        """)
        if file_uploader is not None:
            st.dataframe(file_uploader)
