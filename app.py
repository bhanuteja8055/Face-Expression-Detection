import streamlit as st
import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import tempfile

# Load model
model = load_model('facial_expression_model.h5')
emotion_classes = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# Load Haar cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to detect emotions
def detect_emotion(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    emotions = []
    confidence = {}
    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))
        face = img_to_array(face)
        face = np.expand_dims(face, axis=0) / 255.0

        prediction = model.predict(face)[0]
        emotion_label = emotion_classes[np.argmax(prediction)]
        emotions.append(emotion_label)
        
        # Calculate confidence scores (as percentages)
        for i, score in enumerate(prediction):
            confidence[emotion_classes[i]] = round(score * 100, 2)

    return emotions if emotions else ["No face detected"], confidence if emotions else {}

# Streamlit app
def main():
    st.set_page_config(page_title="Facial Emotion Analyzer", layout="centered")
    
    # Custom CSS for styling
    st.markdown("""
    <style>
    .title {
        font-size: 32px;
        color: #1e3a8a;
        text-align: center;
        margin-bottom: 10px;
    }
    .subtitle {
        font-size: 16px;
        color: #64748b;
        text-align: center;
        margin-bottom: 30px;
    }
    .stButton>button {
        background-color: #3b82f6;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 8px;
        font-size: 16px;
        width: 100%;
        transition: background-color 0.3s;
    }
    .stButton>button:hover {
        background-color: #1e40af;
    }
    .result-box {
        background-color: #f8fafc;
        padding: 20px;
        border-radius: 12px;
        margin-top: 20px;
        text-align: center;
    }
    .confidence-bar {
        background-color: #e5e7eb;
        border-radius: 12px;
        height: 20px;
        margin: 10px 0;
        overflow: hidden;
    }
    .confidence-fill {
        background-color: #3b82f6;
        height: 100%;
        color: #111827;
        font-weight: 600;
        text-align: right;
        padding-right: 5px;
        font-size: 12px;
    }
    </style>
    """, unsafe_allow_html=True)

    # Title and subtitle
    st.markdown('<div class="title">Facial Emotion Analyzer</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Upload an image or use your webcam to detect emotions</div>', unsafe_allow_html=True)

    # Tabs for upload and webcam
    tab1, tab2 = st.tabs(["Upload Image", "Webcam Capture"])

    # Tab 1: Image Upload
    with tab1:
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "bmp"])
        if uploaded_file is not None:
            # Save the uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name

            # Load and process the image
            image = cv2.imread(tmp_file_path)
            if image is not None:
                st.image(image, channels="BGR", caption="Uploaded Image", use_column_width=True)
                if st.button("Analyze Emotion", key="upload_analyze"):
                    emotions, confidence = detect_emotion(image)
                    display_results(emotions, confidence)
            else:
                st.error("Invalid image file.")
            os.unlink(tmp_file_path)  # Clean up temporary file

    # Tab 2: Webcam Capture
    with tab2:
        st.info("Click below to capture an image from your webcam.")
        picture = st.camera_input("Take a picture")
        if picture is not None:
            # Convert the captured image to OpenCV format
            bytes_data = picture.getvalue()
            image = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            if st.button("Analyze Emotion", key="webcam_analyze"):
                emotions, confidence = detect_emotion(image)
                display_results(emotions, confidence)

# Function to display results
def display_results(emotions, confidence):
    st.markdown('<div class="result-box">', unsafe_allow_html=True)
    emoji = {
        "Happy": "😊", "Sad": "😢", "Angry": "😡", "Surprise": "😲",
        "Fear": "😨", "Disgust": "🤢", "Neutral": "😐", "No face detected": "🤔"
    }.get(emotions[0], "🤔")
    st.markdown(f'<h3>{emoji} {emotions[0]}</h3>', unsafe_allow_html=True)
    st.write("Primary detected emotion")

    if confidence:
        st.write("Confidence Levels:")
        for emotion, score in confidence.items():
            st.markdown(f"{emotion}:", unsafe_allow_html=True)
            st.markdown(f'<div class="confidence-bar"><div class="confidence-fill" style="width: {score}%">{score}%</div></div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
