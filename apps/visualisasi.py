# Import library yag digunakan
import streamlit as st
import pandas as pd
from nltk.probability import FreqDist
import matplotlib.pyplot as plt

# Function app untuk page visualisasi
def app():
    
    st.header('Visualisasi Data yang Sering Tampil')
    st.markdown("""
    ### Mengetahui Kata yang Sering Kali Muncul
    """)

    file_uploader = st.file_uploader('Upload Data Preprocessing :', ['csv'])

    if file_uploader is not None:
        data = pd.read_csv(file_uploader)
        df = data
        
        all_tokens = [token for sublist in df['stemmed'] for token in sublist]
        freq_dist = FreqDist(all_tokens)
        
        # Buat objek gambar Matplotlib
        fig, ax = plt.subplots()
        freq_dist.plot(30, cumulative=False)  # Menggunakan parameter cumulative=False
        plt.tight_layout()  # Untuk memastikan tata letak gambar yang baik
        
        # Tampilkan gambar menggunakan st.pyplot()
        st.pyplot(fig)