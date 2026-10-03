import streamlit as st
import cv2
import numpy as np
import pickle
from PIL import Image

class PurePythonKNN:
    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict_proba(self, X):
        probs = []
        for x in X:
            distances = np.linalg.norm(self.X_train - x, axis=1)
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = self.y_train[k_indices]
            
            counts = np.bincount(k_nearest_labels, minlength=4)
            probs.append(counts / self.k)
        return np.array(probs)

st.set_page_config(page_title="Brain Tumor Detector", layout="wide")

st.title("🧠 Brain Tumor Classification & Diagnostic System")
st.write("Upload a Brain MRI scan to analyze tumor categories with feature extraction visualizations.")

CLASS_NAMES = ['Glioma Tumor', 'Meningioma Tumor', 'No Tumor', 'Pituitary Tumor']

@st.cache_resource
def load_model_data():
    with open('brain_tumor_model.pkl', 'rb') as f:
        data = pickle.load(f)
    return data

try:
    data = load_model_data()
    model = data['model']
    
    # Sidebar Metrics
    st.sidebar.header("📊 Model Metrics")
    st.sidebar.metric(label="Overall Test Accuracy", value=f"{data.get('accuracy', 75.75)}%")
    st.sidebar.divider()
    st.sidebar.write(f"**Training Samples:** {data.get('train_count', 5600)}")
    st.sidebar.write(f"**Testing Samples:** {data.get('test_count', 1600)}")
    st.sidebar.write("**Algorithm:** Pure-Python KNN (k=5)")
    st.sidebar.write("**Feature Extraction:** Histogram + Stats")
    
    st.success("Model loaded successfully!")
except Exception as e:
    st.error("Please run 'python train_model.py' first to update the model file.")

uploaded_file = st.file_uploader("Upload Brain MRI Image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and 'model' in locals():
    image = Image.open(uploaded_file).convert('L')
    img_array = np.array(image)
    img_resized = cv2.resize(img_array, (64, 64))

    st.divider()
    st.subheader("🔍 Image Preprocessing & Feature Analysis")

    # Display 3 visual analysis columns
    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("**1. Original MRI Scan**")
        st.image(uploaded_file, use_container_width=True)

    with col2:
        st.write("**2. Tumor Boundary / Edges**")
        # Apply Canny Edge Detection for structural feature view
        edges = cv2.Canny(img_resized, 50, 150)
        st.image(edges, caption="Extracted Tumor Edges", use_container_width=True)

    with col3:
        st.write("**3. Brightness Histogram Graph**")
        hist = cv2.calcHist([img_resized], [0], None, [32], [0, 256]).flatten()
        st.bar_chart(hist)

    # Feature Extraction for Model Prediction
    hist_norm = hist / (hist.sum() + 1e-7)
    stats = np.array([np.mean(img_resized), np.std(img_resized), np.max(img_resized)]) / 255.0
    features = np.hstack([hist_norm, stats]).reshape(1, -1)

    with st.spinner("Analyzing scan..."):
        prediction_prob = model.predict_proba(features)[0]
        predicted_idx = np.argmax(prediction_prob)
        predicted_class = CLASS_NAMES[predicted_idx]
        confidence = prediction_prob[predicted_idx] * 100

    st.divider()
    st.subheader("📋 Classification Result")

    if predicted_class == 'No Tumor':
        st.success(f"**Diagnosis:** {predicted_class} (Confidence: {confidence:.2f}%)")
    else:
        st.error(f"**Diagnosis:** Detected - {predicted_class} (Confidence: {confidence:.2f}%)")

    st.subheader("Probability Breakdown:")
    for idx, name in enumerate(CLASS_NAMES):
        prob = prediction_prob[idx]
        st.progress(float(prob), text=f"{name}: {prob*100:.2f}%")