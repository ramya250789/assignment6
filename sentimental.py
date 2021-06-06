import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
import joblib
from PIL import Image

img = Image.open("tweet.jpg")
st.image(img, width=550)
st.title('Covid tweet Sentimental classification')
senti_model = joblib.load('naive_model_sentimental.joblib')
vectorizer = joblib.load('CountVectorizer_1.joblib')
inp_text = st.text_area('TYPE YOUR TEXT HERE', height=200)
vectorised_text = vectorizer.transform([inp_text])
result = ''


def sentimental_predict(inp):
    prediction = senti_model.predict(inp)
    if prediction == 1:
        pred = 'POSITIVE'
    elif prediction == 2:
        pred = 'NEGATIVE'
    else:
        pred = 'NEUTRAL'
    return pred


if st.button('Submit'):
    result = sentimental_predict(vectorised_text)
    if result == 'POSITIVE':
        st.write("RESULT:", result)
        img = Image.open("positive.jpg")
        st.image(img, width=50)
    elif result == 'NEGATIVE':
        st.write("RESULT:", result)
        img = Image.open("negative.jpg")
        st.image(img, width=50)
    else:
        st.write("RESULT:", result)
        img = Image.open("neutral.jpg")
        st.image(img, width=50)
