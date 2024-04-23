import pandas as pd
import numpy as np
import math
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import streamlit as st

# Function app untuk page visualisasi
def app():
    
    st.header('TF-IDF Calculation')
    st.markdown("""
    ### Mengetahui Kata yang Sering Kali Muncul
    """)

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        df['text_clean'] = df['full_text'].apply(preprocess_text)

        df['tokenize'] = df['text_clean'].apply(lambda x: x.split())
        df['tf'] = df['tokenize'].apply(calculate_tf)

        # Calculate IDF
        documents = df['tokenize'].tolist()
        idf = calculate_idf(documents)

        # Calculate TF-IDF for each document
        df['tf_idf'] = df.apply(lambda row: calculate_tf_idf(row['tf'], idf), axis=1)

        # Display TF-IDF
        st.write(df[['full_text', 'tf_idf']])
        
        name_file = st.sidebar.text_input('Simpan ke Bentuk CSV Data')
        if st.sidebar.button('Simpan'):
            df.to_csv(f'{name_file}.csv')
    else:
        st.warning("Silakan upload file CSV untuk Tahap TF-IDF")

# Function to preprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
    preprocessed_text = ' '.join(lemmatized_tokens)
    return preprocessed_text

# Function to calculate Term Frequency (TF)
def calculate_tf(tokens):
    total_words = len(tokens)
    tf = {}
    for word in tokens:
        tf[word] = tf.get(word, 0) + 1
    for word in tf:
        tf[word] /= total_words
    return tf

# Function to calculate Inverse Document Frequency (IDF)
def calculate_idf(documents):
    N = len(documents)
    idf = {}
    all_words = set(word for document in documents for word in document)
    for word in all_words:
        count = sum(1 for document in documents if word in document)
        idf[word] = math.log10(N / count)
    return idf

# Function to calculate TF-IDF
def calculate_tf_idf(tf, idf):
    tf_idf = {}
    for word in tf:
        tf_idf[word] = tf[word] * idf[word]
    return tf_idf
