import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import binary_crossentropy

conn = sqlite3.connect('feedbacks.db')
c = conn.cursor()

def get_feedbacks():
    c.execute("SELECT * FROM feedbacks")
    feedbacks = c.fetchall()
    return feedbacks

# Create a table to store the feedbacks if it doesn't exist
c.execute('''CREATE TABLE IF NOT EXISTS feedbacks
             (id INTEGER PRIMARY KEY AUTOINCREMENT,
              feedback TEXT,
              sentiment TEXT)''')
conn.commit()




model = load_model("Models/modeldarijaV2.h5")
optimizer = Adam(learning_rate=0.001)
loss = binary_crossentropy
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

# Load the tokenizer from the JSON file
with open('Tokenizer/tokenizerV6.json', 'r', encoding='utf-8') as f:
    tokenizer_json = f.read()
    tokenizer_config = json.loads(tokenizer_json)
    tokenizer = tokenizer_from_json(tokenizer_json)

def add_feedback(feedback, sentiment):
    try:
        # Insert the feedback into the database
        c.execute("INSERT INTO feedbacks (feedback, sentiment) VALUES (?, ?)", (feedback, sentiment))
        conn.commit()
        st.success('Thank you for your feedback!')
    except Exception as e:
        st.error('Error occurred while inserting feedback: {}'.format(e))


# Define the function for sentiment analysis
def sentiment_analysis(text):
    text_sequence = tokenizer.texts_to_sequences([text])
    text_sequence_padded = pad_sequences(text_sequence, maxlen=250)
    prediction = model.predict(text_sequence_padded)[0]
    predicted_label = np.argmax(prediction)
    return predicted_label, prediction[0]

# Define the Streamlit app
def app():
    st.set_page_config(page_title="Analyse de sentiments app",page_icon="http://estfbs.usms.ac.ma/wp-content/uploads/2020/06/cropped-LogoESTFBS-1.png", layout="wide", initial_sidebar_state="auto")
    st.markdown("""
    <style>
    body {
    color: #1B1C1E;
    background-color: #FFFFFF;
    }
    </style>
    """, unsafe_allow_html=True)


    
    # Add a menu with two pages
    pages = ['À propos','Analyse de sentiments','Saved Feedbacks','Nuage de Mots']
    choice = st.sidebar.selectbox('Select a page', pages)
    
    # Display the selected page
    if choice == 'Analyse de sentiments':
        st.title('Analyse de sentiments')
        max_length = st.slider('Maximum text length', min_value=50, max_value=500, value=250, step=50)
        text = st.text_area('Enter some text', max_chars=max_length)
        if st.button('Analyze'):
            result, probability = sentiment_analysis(text)
            emopredict="The predicted sentiment is:"
            st.success(emopredict)
            st.write('Positive 🤗' if result >= 0.5 else 'Negative 😠')
            # Create a bar chart of the predicted probabilities
            df = pd.DataFrame({'Sentiment': ['Positive', 'Negative'], 'Probability': [1-probability, probability]})
            fig = px.bar(df, x='Sentiment', y='Probability', color='Sentiment', text='Probability', height=400)
            fig.update_traces(texttemplate='%{text:.2%}', textposition='inside')
            fig.update_layout(title='Sentiment Analysis Probability', xaxis_title='', yaxis_title='Probability')
            st.write(f'<style>div.row-widget.stHorizontal{ "{" }justify-content: center;{ "}" }</style>', unsafe_allow_html=True)
            st.plotly_chart(fig)
            st.title('Sentiment Analysis Feedback')
            sentiment = st.radio('Sentiment:', ['Positive', 'Negative'])
            if st.button('Feed'):
                add_feedback(text, sentiment)

    elif choice == 'Nuage de Mots':
        
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.title("Nuage de mots")
            user_input = st.text_area('Enter some text', max_chars=500)
            words = [word.lower() for word in TextBlob(user_input).words if word.isalpha()]
            if words:
                word_cloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(words))
                plt.imshow(word_cloud)
                plt.axis("off")
                st.pyplot()

    elif choice == 'Saved Feedbacks':
        st.title('Saved Feedbacks')
        feedbacks = get_feedbacks()
        if feedbacks:
            for feedback in feedbacks:
                st.write("ID: ", feedback[0])
                st.write("Feedback: ", feedback[1])
                st.write("Sentiment: ", feedback[2])
                st.write("-" * 30)
    else :
       
            st.write("")
            st.title("À propos de ce projet")
            st.write("Bienvenue dans notre projet d'analyse des sentiments pour le dialecte marocain DARIJA 🇲🇦 en utilisant le modèle de réseaux de neurones LSTM, inscrit dans notre Stage de fin d'année a l'Ecole Supérieure de technologie de Fqih Ben saleh.")
            st.write("Nous tenons à exprimer notre gratitude envers les professeurs, les encadrants et tous les membres de l'université qui nous ont offert l'opportunité de mener à bien ce projet,leur soutien et leur expertise ont été inestimables pour la réalisation de ce travail.")
            st.write("Ce projet d'analyse des sentiments pour le dialecte marocain DARIJA est une étape importante dans l'exploration de la langue et de la culture marocaines. Les résultats de notre travail pourront être utilisés dans divers domaines tels que le marketing, les études de marché, la politique, etc. De plus, notre modèle de réseaux de neurones LSTM peut être amélioré et étendu pour prendre en charge d'autres dialectes ou langues. Nous espérons que notre projet pourra inspirer et encourager d'autres étudiants à explorer et à exploiter les richesses linguistiques et culturelles de leur pays.")
            


if __name__ == '__main__':
    app()
