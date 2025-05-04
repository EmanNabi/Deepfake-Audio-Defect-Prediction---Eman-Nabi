import streamlit as st
import numpy as np
import librosa 
import joblib
import tensorflow as tf 
from sklearn.preprocessing import StandardScaler
import os

# Title
st.title("🔊 Deepfake Audio & 🐞 Defect Prediction")

# Tabs
tab1, tab2 = st.tabs(["🎤 Audio Detection", "📊 Defect Prediction"])

# Tab 1: Audio Detection
with tab1:
    st.header("Deepfake Audio Detector")
    audio_file = st.file_uploader("Upload WAV file", type=["wav"])
    model_choice = st.selectbox("Select Model", ["SVM", "Logistic Regression", "DNN"])

    if st.button("Analyze Audio"):
        if audio_file:
            try:
                # Extract MFCCs (with same padding as training)
                y, sr = librosa.load(audio_file, sr=None)
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                mfcc = np.pad(mfcc, ((0, 0), (0, 100 - mfcc.shape[1])))[:, :100].flatten().reshape(1, -1)

                # Load model
                if model_choice == "DNN":
                    model = tf.keras.models.load_model("deepfake_dnn.h5")
                    prob = model.predict(mfcc)[0][0]
                else:
                    model = joblib.load(f"{model_choice.lower().replace(' ', '_')}_audio_model.pkl")
                    prob = model.predict_proba(mfcc)[0][1] if hasattr(model, "predict_proba") else model.decision_function(mfcc)[0]

                # Display result
                st.success(f"Prediction: {'Deepfake' if prob > 0.5 else 'Bonafide'} (Confidence: {prob:.2f})")
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.warning("Please upload an audio file.")

# Tab 2: Defect Prediction
with tab2:
    st.header("Multi-Label Defect Predictor")
    input_data = st.text_input("Enter comma-separated features (e.g., 0.5, 1.2, ...):")
    model_choice = st.selectbox("Select Model", ["Logistic Regression", "SVM", "Online Perceptron", "DNN"])

    if st.button("Predict Defects"):
        if input_data:
            try:
                # Parse input
                input_array = np.array([float(x.strip()) for x in input_data.split(",")]).reshape(1, -1)
                
                # Scale features
                scaler = joblib.load("scaler.pkl")
                input_scaled = scaler.transform(input_array)

                # Load model
                if model_choice == "DNN":
                    model = tf.keras.models.load_model("defect_dnn.h5")
                    probs = model.predict(input_scaled)[0]
                    preds = (probs > 0.5).astype(int)
                else:
                    model = joblib.load(f"{model_choice.lower().replace(' ', '_')}_defect_model.pkl")
                    preds = model.predict(input_scaled)[0]
                    probs = model.predict_proba(input_scaled)[0] if hasattr(model, "predict_proba") else preds

                # Display results
                labels = ["Blocker", "Regression", "Bug"]  # Adjust to your labels
                st.success("Predicted Labels: " + ", ".join([labels[i] for i, p in enumerate(preds) if p == 1]))
                st.write("Probabilities:", dict(zip(labels, probs)))
            except Exception as e:
                st.error(f"Invalid input or model error: {e}")
        else:
            st.warning("Please enter feature values.")