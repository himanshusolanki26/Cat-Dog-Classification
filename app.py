import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import os

# =======================================================
# 1) STREAMLIT CONFIG
# =======================================================
st.set_page_config(page_title="Cat–Dog Classification", layout="centered")

MODEL_PATH = "model.keras"

# =======================================================
# 2) MODEL LOADING
# =======================================================
if not os.path.exists(MODEL_PATH):
    st.error(f"Error: Model file '{MODEL_PATH}' not found.")
    st.stop()

@st.cache_resource
def load_keras_model(path):
    return load_model(path, compile=False)

try:
    model = load_keras_model(MODEL_PATH)
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()

# =======================================================
# 3) SIMPLE CLEAN THEME
# =======================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');

html, body, .stApp {
    background: #f9fafb;
    font-family: 'Inter', sans-serif;
    color: #111827;
}

/* Header */
h1 {
    text-align: center;
    font-weight: 800;
    font-size: 42px;
    margin-bottom: 6px;
    background: linear-gradient(90deg, #2563EB, #06B6D4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.subtitle {
    text-align: center;
    font-size: 15px;
    color: #374151;
    margin-bottom: 22px;
}

/* Card */
.card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 14px;
    padding: 22px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.06);
}

/* File uploader */
.stFileUploader > div {
    border: 2px dashed #2563EB;
    border-radius: 12px;
    padding: 18px;
    background: #f9fafb;
    margin-bottom: 20px;
}

/* Prediction label */
.prediction-label {
    font-weight: 700;
    font-size: 24px;
    color: #2563EB;
    margin-top: 10px;
}
.confidence-chip {
    display: inline-block;
    margin-top: 8px;
    padding: 6px 12px;
    border-radius: 999px;
    background: #ecfdf5;
    color: #065f46;
    font-weight: 600;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

# =======================================================
# 4) HEADER
# =======================================================
st.markdown("<h1>Cat–Dog Classification</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Upload an image to see if it’s a cat or a dog.</p>", unsafe_allow_html=True)

# =======================================================
# 5) ONE-PAGE APP
# =======================================================
st.markdown("<div class='card'>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')

    # Preprocess
    input_tensor_shape = model.input_shape
    input_shape = input_tensor_shape[1:3] if len(input_tensor_shape) == 4 else (224, 224)
    img_resized = img.resize((input_shape[0], input_shape[1]))
    img_array = image.img_to_array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    with st.spinner("Analyzing..."):
        prediction = model.predict(img_array, verbose=0)

    CAT_LABEL = "Cat"
    DOG_LABEL = "Dog"

    if prediction.shape[1] == 1:
        pred_prob = float(prediction[0][0])
        pred_class = DOG_LABEL if pred_prob >= 0.5 else CAT_LABEL
        prob_dict = {DOG_LABEL: pred_prob * 100, CAT_LABEL: (1 - pred_prob) * 100}
    elif prediction.shape[1] == 2:
        pred_prob = prediction[0]
        pred_class = DOG_LABEL if np.argmax(pred_prob) == 1 else CAT_LABEL
        prob_dict = {CAT_LABEL: float(pred_prob[0]) * 100, DOG_LABEL: float(pred_prob[1]) * 100}
    else:
        st.error("Model output not standard (expects 1 or 2 classes).")
        st.stop()

    # Create two columns: left (image), right (results)
    col1, col2 = st.columns([1, 2])  # adjust ratio as needed

    with col1:
        st.image(img, caption="Uploaded Image", use_column_width=True)

    with col2:
        # Prediction
        st.markdown(f"<div class='prediction-label'>{pred_class}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='confidence-chip'>Confidence: {prob_dict[pred_class]:.2f}%</div>", unsafe_allow_html=True)

        # Probability chart
        fig = go.Figure(go.Bar(
            x=[prob_dict[CAT_LABEL], prob_dict[DOG_LABEL]],
            y=[CAT_LABEL, DOG_LABEL],
            orientation='h',
            marker=dict(color=["#06B6D4", "#2563EB"])
        ))
        fig.update_layout(
            title="Probability Distribution",
            title_font_color="#111827",
            xaxis_title="Percentage (%)",
            yaxis_title="",
            showlegend=False,
            plot_bgcolor='rgba(255,255,255,0)',
            paper_bgcolor='rgba(255,255,255,0)',
            font=dict(color="#111827", size=14),
            margin=dict(l=10, r=10, t=35, b=10),
            xaxis=dict(gridcolor='rgba(0,0,0,0.05)')
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

else:
    st.write("Please upload an image to classify.")

st.markdown("</div>", unsafe_allow_html=True)
