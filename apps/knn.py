import streamlit as st
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def app():
    st.sidebar.subheader('Inputan')
    st.sidebar.number_input('Masukkan Size Untuk Data Training :')
    st.sidebar.button('Tampil Data')
    
    st.sidebar.subheader('Sentiment Analysis :')
    
    st.sidebar.subheader('Split Data :')
    st.sidebar.subheader('Data Traing : (62616) / Data Testing : (76478)')
    st.sidebar.subheader('Jumlah Data : 654')
    
    st.header('K-NearestNeighbors')
    
    file_uploader = st.file_uploader('Upload Data CSV :', ['csv'])

    
