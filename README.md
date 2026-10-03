# 🧠 Brain Tumor Classification & Diagnostic System

An interactive end-to-end Computer Vision and Machine Learning web application built using Python, OpenCV, and Streamlit. The system analyzes Brain MRI scans, extracts intensity and structural features, and classifies tumor categories in real-time.

---

## 🌟 Key Features

* **Multi-Column Visual Pipeline:** Displays original MRI scans, Canny edge boundary extractions, and pixel intensity brightness histograms simultaneously.
* **Custom Feature Extraction:** Utilizes normalized intensity histograms and statistical metrics (Mean, Std Dev, Max Brightness) instead of raw pixel flattening for robust classification.
* **Pure Python Machine Learning Engine:** Implements a custom $k$-Nearest Neighbors ($k=5$) algorithm tailored for high-dimensional feature vectors.
* **Performance Dashboard:** Displays real-time test accuracy, total sample counts, and algorithm parameters on a dedicated sidebar.
* **Probability Breakdown:** Provides exact percentage confidence metrics across 4 target classes:
  * Glioma Tumor
  * Meningioma Tumor
  * Pituitary Tumor
  * No Tumor

---

## 🛠️ Tech Stack

* **Language:** Python
* **Web Framework:** Streamlit
* **Computer Vision:** OpenCV (`cv2`), Pillow
* **Numerical Processing:** NumPy
* **Model Serialization:** Pickle

---

## 📁 Project Structure

```text
Brain-Tumor-Classification-Streamlit/
├── Dataset/
│   ├── Testing/
│   └── Training/
├── app.py
├── train_model.py
├── brain_tumor_model.pkl
├── README.md
└── .gitignore