# Import library yang digunakan
import streamlit as st
from multiapp import MultiApp
from apps import preprocessing_data, tf_idf, knn

app = MultiApp()

# Headline aplikasi
st.markdown("""
# Analisis Sentiment Twitter dengan Metode K-NearestNeigbors

Ini adalah aplikasi yang berfungsi untuk menganalisis sentiment data twitter beserta klasifikasinya

""")

st.sidebar.header('User Input Features')

# Pemanggilan function untuk setiap page
app.add_app("Preprocessing Data", preprocessing_data.app)
app.add_app("TF-IDF", tf_idf.app)
app.add_app("K-NearestNeighbors", knn.app)

app.run()
