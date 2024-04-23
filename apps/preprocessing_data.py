# Import library yang digunakan
import os
import streamlit as st
import pandas as pd
import re
import string
import nltk
from nltk.tokenize import word_tokenize
# nltk.download('all')
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Function app untuk page preprocessing
def app():
    # Bagian Sidebar
    st.sidebar.write('Data yang Dibersihkan')
    st.sidebar.checkbox('Remove Username (@)', value=True)
    st.sidebar.checkbox('Remove ReTweet & Hastag', value=True)
    st.sidebar.checkbox('Remove URL or http', value=True)
    st.sidebar.checkbox('Remove Simbol & Number', value=True)
    st.sidebar.checkbox('Remove Duplicate', value=True)
    st.sidebar.checkbox('Lower Case', value=True)
    st.sidebar.checkbox('Tokenizing', value=True)
    st.sidebar.checkbox('Remove Stopwords', value=True)
    st.sidebar.checkbox('Stemming', value=True)
    
    st.header('Preprocessing Data')
    st.subheader("oke")
    file_uploader = st.file_uploader('Upload Data CSV :', ['csv'])
    
    if file_uploader is not None:
        data = pd.read_csv(file_uploader)
        df = data
        
        df = df[['full_text']]
        df = df.drop_duplicates(subset=['full_text'])
        df = df.dropna()
        
        df['text_clean'] = df['full_text'].apply(preprocessing_text)
        df['tokenize'] = df['text_clean'].apply(tokenize_text)
        df['stopwords'] = df['tokenize'].apply(stopword_text)
        df['stemmed'] = df['stopwords'].apply(stemming_text)
        
        data_preprocessed = df.head(10)

        df.to_csv('hasil_preprocessing.csv', index=False)
        
        col1, col2 = st.columns(2)

        # Kolom untuk tabel original text dan processed text
        with col1:
            st.markdown("""
            ###### Original Text
            """)
            if file_uploader is not None:
                dataframe1 = pd.DataFrame(data.head(10))
                st.table(dataframe1)
        with col2:
            st.markdown("""
            ###### Processed Text
            """)
            if file_uploader is not None:
                dataframe2 = pd.DataFrame(data_preprocessed)
                st.table(dataframe2)
    else:
        st.warning("Silakan upload file CSV untuk Data Cleaning")
            

# Kumpulan Function-Function

# Cleaning Data
def preprocessing_text(kalimat):
    lower_case = kalimat.lower()
    hasil = re.sub(r"\d+", "", lower_case)
    hasil = hasil.translate(str.maketrans("", "", string.punctuation))
    hasil = hasil.strip()
    
    return hasil

# Tokenizing
def tokenize_text(kalimat):
    tokens = nltk.tokenize.word_tokenize(kalimat)
    
    return tokens

# Stopword
def stopword_text(tokens):
    factory = StopWordRemoverFactory()
    stopwords = factory.get_stop_words()
    cleaned_tokens = []
    for token in tokens:
        if token not in stopwords:
            cleaned_tokens.append(token)
            
    return cleaned_tokens

# Stemming
def stemming_text(tokens):
    stem_factory = StemmerFactory()
    stemmer = stem_factory.create_stemmer()
    hasil = [stemmer.stem(token) for token in tokens]
    
    return hasil
