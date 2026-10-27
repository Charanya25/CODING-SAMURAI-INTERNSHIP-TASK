
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Facial Expression Recognition",
    page_icon="😊"
)

st.title("Facial Expression Recognition")
st.write("Upload a face image to predict its visible expression.")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("facial_expression_model.keras")

try:
    model = load_model()
except Exception as e:
    st.error("Model not found. Keep the model file beside app.py.")
    st.stop()

class_names = [
    "angry", "disgust", "fear", "happy",
    "neutral", "sad", "surprise"
]

uploaded_file = st.file_uploader(
    "Choose a face image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")
    st.image(image, caption="Uploaded image", width=300)

    image = image.resize((48, 48))
    image_array = np.array(image, dtype=np.float32) / 255.0
    image_array = image_array.reshape(1, 48, 48, 1)

    predictions = model.predict(image_array, verbose=0)[0]
    predicted_index = int(np.argmax(predictions))

    st.subheader(
        "Predicted expression: "
        + class_names[predicted_index].capitalize()
    )
    st.write(
        f"Model confidence: {predictions[predicted_index] * 100:.2f}%"
    )

    st.caption(
        "This is an educational model. It may make mistakes and "
        "does not determine a person's actual feelings."
    )
