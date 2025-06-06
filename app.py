import base64
import pandas as pd
import numpy as np
import streamlit as st
import requests
import json
import datetime
import pickle
import string
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Alata&family=Funnel+Sans:ital,wght@0,300..800;1,300..800&display=swap');

    body {{ font-family: 'Funnel Sans', sans-serif; }}

    div[class*="stTextArea"] {
        background-color: transparent !important;
    }

    [data-testid="stAppViewContainer"] {
        background-color: #D3D3D3;
    }

    div {
        color: black;
    }

    .stButton > button {
        background-color: black;
        color: white !important;
        border-radius: 20%;
        height: 50px;
        width: 50px;
        font-size: 20px;
        text-align: right;
        margin-top: 12px;
    }

    [data-testid="stTextArea"] {
        background-color: transparent !important;
    }

    /* Wrapper baseweb textarea */
    [data-baseweb="textarea"] {
        background-color: transparent !important;
    }

    /* Elemen textarea */
    [data-baseweb="textarea"] > textarea {
        background-color: transparent !important;
        border-color: transparent !important;
        color: black !important;
        border: 1px solid #ffffff !important;
        resize: none;
    }

    /* Wrapper input */
    [data-baseweb="input"] > div {
        background-color: transparent !important;
    }
            
    }
    </style>
""", unsafe_allow_html=True)

def set_background_by_emotion(emotion):
    color_map = {
        "joy": "#FFE173",
        "sad": "#A0C4FF",
        "anger": "#FFADAD",
        "fear": "#E8C2FF",
        "love": "#F7B5E1",
        "surprise": "#B5F7D6",
        "neutral": "#D3D3D3", 
    }
    bg_color = color_map.get(emotion, "#FFFFFF")

    st.markdown(f"""
        <style>
        [data-testid="stAppViewContainer"] {{
            background-color: {bg_color} !important;
        }}
        </style>
    """, unsafe_allow_html=True)

emotion_images = {
    "joy": "Assets/joy.png",
    "sad": "Assets/sad.png",
    "anger": "Assets/mad.png",
    "fear": "Assets/fear.png",
    "love": "Assets/love.png",
    "surprise": "Assets/suprise.png",
}

spotify_urls = {
    "joy": "https://open.spotify.com/embed/playlist/37i9dQZF1DXdPec7aLTmlC?utm_source=generator",
    "sad": "https://open.spotify.com/embed/playlist/37i9dQZF1EIfSlYfSw82ow?utm_source=generator",
    "anger": "https://open.spotify.com/embed/playlist/37i9dQZF1EIgNZCaOGb0Mi?utm_source=generator",  
    "fear": "https://open.spotify.com/embed/playlist/37i9dQZF1EIfMwRYymgnLH?utm_source=generator",
    "love": "https://open.spotify.com/embed/playlist/37i9dQZF1EIcE10fGQVrZK?utm_source=generator",
    "surprise": "https://open.spotify.com/embed/playlist/2hy4XF1ionrru3deaA3e0m?utm_source=generator"
}

st.markdown("""
    <style>
    .custom-title {
        font-size: 45px;
        font-weight: bold;
        text-align: center;
    }
            
    .custom-sub-title {
        margin-top: -10px;   
        font-size: 15px;
        text-align: center;
    }
    </style>
    
    <div class="custom-title">Emotion Detection App</div>
    <div class="custom-sub-title">This app detects emotions from text using a pre-trained model.</div>
""", unsafe_allow_html=True)

st.markdown("""
    <style>     
    .custom-subtitle {
        margin-top: 15px;   
        font-size: 15px;
        text-align: center;
    }
    </style>
            
    <div class="custom-subtitle">Enter your text:</div>
""", unsafe_allow_html=True)

st.session_state.current_emotion = "neutral"
emotion = 'neutral'
now = datetime.datetime.now()

@st.cache_resource
def load_model():
    with open("naivebayes_emotion_model.pkl", "rb") as f:
        return pickle.load(f)

classifier = load_model()

def get_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def preprocess(text):
  stemmer = SnowballStemmer('english')
  lemmatizer = WordNetLemmatizer()

  tokens = word_tokenize(text)

  tokens = [word.lower() for word in tokens]
  tokens = [word for word in tokens if word not in string.punctuation]
  tokens = [word for word in tokens if word not in stopwords.words('english')]
  tokens = [stemmer.stem(word) for word in tokens]
  tag = pos_tag(tokens)
  tokens = [lemmatizer.lemmatize(word, get_pos(t)) for (word,t) in tag]

  return tokens

def extract_features(text):
  features = {}

  for word in fd.keys():
    features[word] = (word in text)

  return features

col1, col2 = st.columns([10, 1])
with col1:
    text_input = st.text_area('Input Text', key='textarea', height=68, label_visibility="collapsed")

with col2:

    if st.button("➤", key='button'):
        if text_input:
            tokens = preprocess(text_input)
            fd = FreqDist(tokens)
            features = extract_features(tokens)
            emotion = classifier.classify(features)

            label_map = {
                "suprise": "surprise"
            }

            # setelah prediksi
            emotion = label_map.get(emotion, emotion)

            st.session_state.current_emotion = emotion
            
            set_background_by_emotion(emotion)

        else:
            st.write('Please enter some text to analyze.')

if st.session_state.current_emotion:
    st.markdown(f"""
        <div style="text-align: center;">
            <h3>Current Detected Emotion:</h3>
        </div>
    """, unsafe_allow_html=True)

    img_path = emotion_images.get(emotion)
    if img_path:
        st.markdown(
        f"""
        <div style="text-align: center;">
            <img src="data:image/png;base64,{base64.b64encode(open(img_path, "rb").read()).decode()}" width="300"/>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
        <div style="text-align: center;">
            <h1 style="margin-top: 0; padding: 0;">{st.session_state.current_emotion.capitalize()}</h1>
            <p style="color: black; font-size: 14px;">{now.strftime('%I:%M %p')}</p>
        </div>
    """, unsafe_allow_html=True)

    spotify_url = spotify_urls.get(emotion)
    if spotify_url:
        st.markdown(f"""
            <div style="text-align: center;">
                <h3>Playlist for you: </h3>
            </div>
        """, unsafe_allow_html=True)

        st.markdown(
            f"""
            <div style="display: flex; justify-content: center;">
                <iframe style="border-radius:12px" 
                        src="{spotify_url}" 
                        width="100%" 
                        height="152" 
                        frameBorder="0" 
                        allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture" 
                        loading="lazy">
                </iframe>
            </div>
            """, unsafe_allow_html=True)