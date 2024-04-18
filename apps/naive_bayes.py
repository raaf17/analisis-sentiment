# Import library ynag digunakan
import streamlit as st
import pandas as pd
import preprocessor as p
from textblob import TextBlob
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import random
from textblob.classifiers import NaiveBayesClassifier
nltk.download('punkt')

# Function app untuk page naive_bayes
def app():
    # Menu dan Tampilan data pada sidebar
    st.sidebar.subheader('Inputan')
    st.sidebar.number_input('Masukkan Size Untuk Data Training :')
    st.sidebar.button('Tampil Data')
    
    st.sidebar.subheader('Sentiment Analysis :')
    
    st.sidebar.subheader('Split Data :')
    st.sidebar.subheader('Data Traing : (62616) / Data Testing : (76478)')
    st.sidebar.subheader(f'Jumlah Data : 654')
    
    st.header('Naive Bayes')
    
    file_uploader = st.file_uploader('Upload Data CSV :', ['csv'])
    
    if file_uploader is not None:
        data = pd.read_csv(file_uploader)
        
        data_tweet = list(data['tweet_english'])
        polaritas = 0

        status = []
        total_positif = total_negatif = total_netral = total = 0

        for i, tweet in enumerate(data_tweet):
            analysis = TextBlob(tweet)
            polaritas += analysis.polarity

            if analysis.sentiment.polarity > 0.0:
                total_positif += 1
                status.append('Positif')
            elif analysis.sentiment.polarity == 0.0:
                total_netral += 1
                status.append('Netral')
            else:
                total_negatif += 1
                status.append('Negatif')

            total += 1
            
        col1, col2 = st.columns(2)

        # Kolom untuk tampilan visualisasi dan tampilan data deskripsi
        with col1:
            st.markdown("""
            ##### Visualisasi Analysis Tweets
            """)
            sns.set_theme()

            labels = ['Positif', 'Negatif', 'Netral']
            counts = [total_positif, total_negatif, total_netral]

            show_bar_chart(labels, counts, "Distribusi Sentimen Twitter")
        with col2:
            st.markdown("""
            ##### Deskripsi
            """)
            st.text(f'Positif = {total_positif}')
            st.text(f'Netral = {total_netral}')
            st.text(f'Negatif = {total_negatif}')
            st.text(f'Total Data = {total}') 
        
        data['klasifikasi'] = status
        
        # Visualisasi Wordcloud
        st.markdown("""
        #### Visualisasi Wordcloud
        """)
        
        all_words = ' '.join([tweets for tweets in data['full_text']])

        wordcloud = WordCloud(
            width=3000,
            height=2000,
            random_state=3,
            background_color='black',
            colormap='Blues_r',
            collocations=False,
            stopwords=STOPWORDS
        ).generate(all_words)

        plot_cloud(wordcloud)
        
        # Data Training
        dataset = data.drop(['full_text'], axis=1, inplace=False)
        dataset = [tuple(x) for x in dataset.to_records(index=False)]

        set_positif = []
        set_negatif = []
        set_netral = []

        for n in dataset:
            if(n[1] == 'Positif'):
                set_positif.append(n)
            elif(n[1] == 'Negatif'):
                set_negatif.append(n)
            else:
                set_netral.append(n)
                
        set_positif = random.sample(set_positif, k=int(len(set_positif)/2))
        set_negatif = random.sample(set_negatif, k=int(len(set_negatif)/2))
        set_netral = random.sample(set_netral, k=int(len(set_netral)/2))

        train = set_positif + set_negatif + set_netral

        train_set = []

        for n in train:
            train_set.append(n)
            
        cl = NaiveBayesClassifier(train_set)
        st.markdown("""
        #### Akurasi Test
        """)
        st.text(f'Akurasi Test : {cl.accuracy(dataset)}')
        
        data_tweet = list(data['tweet_english'])
        polaritas = 0

        status = []
        total_posistif = total_negatif = total_netral = total = 0

        for i, tweet in enumerate(data_tweet):
            analysis = TextBlob(tweet, classifier=cl)

            if analysis.classify() == 'Positif':
                total_posistif += 1
            elif analysis.classify() == 'Netral':
                total_netral += 1
            else:
                total_negatif += 1

            status.append(analysis.classify())
            total += 1
            
        st.markdown("""
        ##### Hasil Analisis Data
        """)
        st.text(f'Positif = {total_positif}')
        st.text(f'Netral = {total_netral}')
        st.text(f'Negatif = {total_negatif}')
        st.text(f'Total Data = {total}')
        
        status = pd.DataFrame({'klasifikasi_bayes': status})
        data['klasifikasi_bayes'] = status

        data_tweet = list(data['tweet_english'])
        polaritas = 0

        # Testing
        st.markdown("""
        ##### Testing
        """)
        input_text = st.text_input('Masukkan Teks yang Ingin Diuji :')
        hasil_klasifikasi = classify_text(input_text, cl)
        st.text(f'\nHasil Klasifikasi untuk Text yang Dimasukkan : {hasil_klasifikasi}')
        st.success(hasil_klasifikasi)
        

# Kumpulan Function-Function

# Visualisasi Bar Chart
def show_bar_chart(labels, counts, title):
    # fig = plt.figure()
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(labels, counts, color=['#2394f7', '#f72323', '#fac343'])

    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.annotate(f'{count}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.set_xlabel('Sentimen')
        ax.set_ylabel('Jumlah')
        ax.set_title(title)
    st.pyplot(fig)
        
# Visualisasi Wordcloud
def plot_cloud(wordcloud):
    fig = plt.figure(figsize=(10, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()
    st.pyplot(fig)
    
def classify_text(text, cl):
    analysis = TextBlob(text, classifier=cl)
    return analysis.classify()
