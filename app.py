# Import library yang digunakan
import streamlit as st
from multiapp import MultiApp
from apps import naive_bayes, preprocessing_data, visualisasi

app = MultiApp()

# Headline aplikasi
st.markdown("""
# Analisis Sentiment Twitter dengan Metode Naive Bayes

Ini adalah aplikasi yang berfungsi untuk menganalisis sentiment data twitter beserta klasifikasinya

""")

st.sidebar.header('User Input Features')

# Pemanggilan function untuk setiap page
app.add_app("Preprocessing Data", preprocessing_data.app)
app.add_app("Visualisasi", visualisasi.app)
app.add_app("Naive Bayes", naive_bayes.app)

app.run()
