import streamlit as st
from multiapp import MultiApp
from apps import preprocessing_data, labeling, knn # import your app modules here

app = MultiApp()

st.markdown("""
# Analisis Sentiment Twitter dengan Metode K-NearestNeighbors

Ini adalah aplikasi yang berfungsi untuk menganalisis sentiment data twitter beserta klasifikasinya

""")

st.sidebar.header('User Input Features')

# Add all your application here
app.add_app("Preprocessing Data", preprocessing_data.app)
app.add_app("Labeling", labeling.app)
app.add_app("K-NearestNeighbors", knn.app)
# The main app
app.run()
