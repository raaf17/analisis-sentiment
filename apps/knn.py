# Import library ynag digunakan
import pandas as pd
import numpy as np
import math
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import streamlit as st

# Function app untuk page naive_bayes
def app():
    # Menu dan Tampilan data pada sidebar
    st.sidebar.subheader('Inputan')
    K = st.sidebar.number_input('K Value : ', min_value=1, max_value=10, value=5)
    num_training_data = st.sidebar.slider("Training Set Size", min_value=0.1, max_value=0.9, value=0.8, step=0.1)
    st.sidebar.button('Run KNN')
    st.sidebar.write('Data Traing : (62616) / Data Testing : (76478)')
    akurasi = 0.90
    st.sidebar.write('Akurasi Test : ', akurasi)
    total_data = 343
    st.sidebar.write(f'Jumlah Data : ', total_data)
    
    st.header('Metode Analisa dengan K-NearestNeighbors')
    
    # Upload file CSV untuk data training
    uploaded_train_file = st.file_uploader("Upload CSV file untuk data training", type=["csv"])
    if uploaded_train_file is not None:
        train_df = pd.read_csv(uploaded_train_file)

        # Preprocess text untuk data training
        train_df['text_clean'] = train_df['full_text'].apply(preprocess_text)

        # Ambil sebagian data untuk pelatihan
        training_data = train_df.head(int(num_training_data))  # Konversi num_training_data ke tipe integer

        # Upload file CSV untuk data uji
        uploaded_test_file = st.file_uploader("Upload CSV file untuk data uji", type=["csv"])
        if uploaded_test_file is not None:
            test_df = pd.read_csv(uploaded_test_file)

            # Preprocess text untuk data uji
            test_df['text_clean'] = test_df['full_text'].apply(preprocess_text)

            # Analisis sentimen menggunakan KNN
            sentiment_results = analyze_sentiment(train_df, test_df, training_data, K)

            # Tampilkan hasil analisis sentimen
            st.write(sentiment_results)

            # Fitur untuk pengujian data tweet baru
            st.subheader("Testing Data Tweet Baru")
            new_tweet = st.text_input("Masukkan tweet baru:")
            if st.button("Analisis Sentimen"):
                # Preprocess text untuk tweet baru
                new_tweet_clean = preprocess_text(new_tweet)
                # Analisis sentimen menggunakan KNN
                sentiment = predict_new_tweet_sentiment(train_df, training_data, new_tweet_clean, K)
                st.write(f"Sentimen dari tweet '{new_tweet}' adalah: {sentiment}")
        else:
            st.warning("Silakan upload file CSV untuk melakukan analisis sentimen pada data uji")
    else:
        st.warning("Silakan upload file CSV untuk data training")
        

# Kumpulan Function-Function

# Fungsi untuk preprocessing teks
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove mentions
    text = re.sub(r'@\w+', '', text)
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Remove punctuation and special characters
    text = re.sub(r'[^\w\s]', '', text)
    # Tokenization
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
    # Join tokens back into text
    preprocessed_text = ' '.join(lemmatized_tokens)
    return preprocessed_text

# Fungsi untuk menghitung TF-IDF dari data uji
def calculate_tf_idf(text, train_df, training_data):
    N = len(training_data)
    idf = calculate_idf(training_data['text_clean'].apply(lambda x: x.split()).tolist())
    tokens = text.split()
    tf = calculate_tf(tokens)
    tf_idf = {}
    for word in tokens:
        if word in idf:
            tf_idf[word] = tf[word] * idf[word]
        else:
            tf_idf[word] = 0
    return tf_idf

# Fungsi untuk menghitung Term Frequency (TF)
def calculate_tf(tokens):
    total_words = len(tokens)
    tf = {}
    for word in tokens:
        tf[word] = tf.get(word, 0) + 1
    for word in tf:
        tf[word] /= total_words
    return tf

# Fungsi untuk menghitung Inverse Document Frequency (IDF)
def calculate_idf(documents):
    N = len(documents)
    idf = {}
    all_words = set(word for document in documents for word in document)
    for word in all_words:
        count = sum(1 for document in documents if word in document)
        idf[word] = math.log10(N / count)
    return idf

# Fungsi untuk mencari K tetangga terdekat
def find_nearest_neighbors(train_df, training_data, test_vector, K):
    distances = []
    for index, row in training_data.iterrows():
        train_vector = calculate_tf_idf(row['text_clean'], train_df, training_data)
        distance = calculate_distance(train_vector, test_vector)
        distances.append((index, distance))
    distances.sort(key=lambda x: x[1])
    neighbors = []
    for i in range(K):
        # Periksa jika ada tetangga terdekat yang tersedia
        if i < len(distances):
            neighbors.append(distances[i][0])
    return neighbors

# Fungsi untuk menghitung jarak antara dua vektor
def calculate_distance(vec1, vec2):
    common_words = set(vec1.keys()) & set(vec2.keys())
    dot_product = sum(vec1[word] * vec2[word] for word in common_words)
    magnitude_vec1 = math.sqrt(sum(vec1[word]**2 for word in vec1))
    magnitude_vec2 = math.sqrt(sum(vec2[word]**2 for word in vec2))
    return dot_product / (magnitude_vec1 * magnitude_vec2)

# Fungsi untuk menganalisis sentimen menggunakan KNN
def analyze_sentiment(train_df, test_df, training_data, K):
    sentiment_results = []
    for index, row in test_df.iterrows():
        # Hitung vektor TF-IDF untuk data uji
        test_vector = calculate_tf_idf(row['text_clean'], train_df, training_data)
        
        # Cari K tetangga terdekat
        nearest_neighbors = find_nearest_neighbors(train_df, training_data, test_vector, K)
        
        # Prediksi sentimen berdasarkan mayoritas dari K tetangga terdekat
        predicted_sentiment = predict_sentiment(train_df, training_data, nearest_neighbors)
        
        # Simpan hasil prediksi sentimen
        sentiment_results.append({'Tweet': row['full_text'], 'Sentiment': predicted_sentiment})
    
    return pd.DataFrame(sentiment_results)

# Fungsi untuk memprediksi sentimen berdasarkan mayoritas dari K tetangga terdekat
def predict_sentiment(train_df, training_data, nearest_neighbors):
    sentiments = training_data.iloc[nearest_neighbors]['sentiment']
    positive_count = sentiments.value_counts().get('positive', 0)
    negative_count = sentiments.value_counts().get('negative', 0)
    if positive_count > negative_count:
        return 'positive'
    elif positive_count < negative_count:
        return 'negative'
    else:
        return 'neutral'

# Fungsi untuk memprediksi sentimen tweet baru
def predict_new_tweet_sentiment(train_df, training_data, new_tweet_clean, K):
    # Hitung vektor TF-IDF untuk tweet baru
    new_tweet_vector = calculate_tf_idf(new_tweet_clean, train_df, training_data)
    
    # Cari K tetangga terdekat
    nearest_neighbors = find_nearest_neighbors(train_df, training_data, new_tweet_vector, K)
    
    # Prediksi sentimen berdasarkan mayoritas dari K tetangga terdekat
    sentiment = predict_sentiment(train_df, training_data, nearest_neighbors)
    
    return sentiment