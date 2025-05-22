# app.py (Streamlit app)

import streamlit as st
import joblib

# Load saved model and vectorizer
model = joblib.load('spam_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

st.title("📧 Email/SMS Spam Classifier")

user_input = st.text_area("Enter your email or SMS message here:")

if st.button("Predict"):
    if user_input.strip():
        # Vectorize input and predict
        vect_input = vectorizer.transform([user_input])
        prediction = model.predict(vect_input)[0]
        label = "Spam 🚫" if prediction == 1 else "Not Spam ✅"
        st.success(f"Prediction: **{label}**")
    else:
        st.warning("Please enter some text to classify.")
