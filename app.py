# app.py

import streamlit as st
import pickle
import string
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# 🔧 Download required NLTK resources if missing
nltk.data.path.append('./nltk_data')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Initialize stemmer and stopwords
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

# 🧹 Preprocessing function
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stop_words:
            y.append(ps.stem(i))

    return " ".join(y)

# 📦 Load the model and vectorizer using safe paths
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')
VECTORIZER_PATH = os.path.join(os.path.dirname(__file__), 'vectorizer.pkl')

try:
    tfidf = pickle.load(open(VECTORIZER_PATH, 'rb'))
    model = pickle.load(open(MODEL_PATH, 'rb'))
except Exception as e:
    st.error(f"Error loading model/vectorizer: {e}")
    st.stop()

# 📱 Streamlit UI
st.title("📩 SMS Spam Classifier")
st.markdown("### A simple spam detector trained by Shekhar")

# 📝 Text input
input_sms = st.text_area("Enter the SMS message")

# 🔍 Predict button
if st.button('Predict'):
    if input_sms.strip() == "":
        st.warning("Please enter a message.")
    else:
        # 1. Preprocess
        transformed_sms = transform_text(input_sms)
        # 2. Vectorize
        vector_input = tfidf.transform([transformed_sms])
        # 3. Predict
        result = model.predict(vector_input)[0]
        # 4. Display
        if result == 1:
            st.error("🚫 Spam")
        else:
            st.success("✅ Not Spam")
