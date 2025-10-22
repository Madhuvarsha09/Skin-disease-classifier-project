import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="Skin Disease Classifier", page_icon="🩺", layout="centered")

st.markdown("<h1 style='text-align:center;'>🌸 Skin Disease Classifier</h1>", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model("models/model.h5")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

uploaded_file = st.file_uploader("Upload a skin image...", type=["jpg", "jpeg", "png"])

classes = ['Eczema', 'Melanoma', 'Acne', 'Psoriasis', 'Rosacea', 'Healthy Skin']

def predict_image(image):
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)
    class_index = np.argmax(preds)
    confidence = float(np.max(preds))
    return classes[class_index], confidence, preds

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("🔍 Predict", use_container_width=True):
        pred_class, conf, all_preds = predict_image(image)

        st.markdown(f"<h3 style='text-align:center;'>Prediction: {pred_class}</h3>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align:center;'>Confidence: <b>{conf*100:.2f}%</b></p>", unsafe_allow_html=True)

        # Show probability distribution
        st.subheader("Class Probabilities:")
        for i, c in enumerate(classes):
            st.write(f"{c}: {all_preds[0][i]*100:.2f}%")

st.markdown("""
<style>
.stButton>button {
    background-color:#007bff;
    color:white;
    border-radius:10px;
    font-size:16px;
    padding:0.6em 1.4em;
    transition:0.3s;
}
.stButton>button:hover {
    background-color:#0056b3;
    transform:scale(1.03);
}
</style>
""", unsafe_allow_html=True)
