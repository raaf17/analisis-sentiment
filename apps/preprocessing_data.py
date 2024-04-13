import streamlit as st
import pandas as pd

def app():
    st.sidebar.checkbox('Remove Username (@)', value=True)
    st.sidebar.checkbox('Remove ReTweet & Hastag', value=True)
    st.sidebar.checkbox('Remove URL or http', value=True)
    st.sidebar.checkbox('Remove Simbol & Number', value=True)
    st.sidebar.checkbox('Remove Duplicate', value=True)
    st.sidebar.checkbox('Lower Case', value=True)
    st.sidebar.checkbox('Tokenizing', value=True)
    st.sidebar.checkbox('Remove Stopwords', value=True)
    st.sidebar.checkbox('Lemmatizing or Stemming', value=True)
    st.sidebar.checkbox('Join Case', value=True)
    
    st.header('Preprocessing Data')

    file_uploader = st.file_uploader('Upload Data CSV :', ['csv'])
    if file_uploader is not None:
        dataupload = pd.read_csv(file_uploader)
        
    
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ###### Original Text
        """)
        if file_uploader is not None:
            st.dataframe(dataupload)
    with col2:
        st.markdown("""
        ###### Processed Text
        """)
        if file_uploader is not None:
            st.dataframe(dataupload)
