import os
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers, models
from PIL import Image

# Class names in the same order as training directories
CLASS_NAMES = [
    "American skunk cabbage",
    "Chilean rhubarb",
    "Curly waterweed",
    "Floating pennywort",
    "Giant hogweed",
    "Himalayan balsam",
    "Non-invasive",
    "Nuttalls waterweed",
    "Parrots feather",
]

@st.cache_resource
def load_model():
    # Load feature extractor from bundled directory
    feature_path = os.path.join(
        os.path.dirname(__file__),
        "inception-v3-tensorflow2-inaturalist-inception-v3-feature-vector-v2",
    )
    feature_extractor_model = tf.saved_model.load(feature_path)

    class FeatureExtractor(layers.Layer):
        def __init__(self, model):
            super().__init__()
            # Keep a reference to the full SavedModel so that the
            # underlying variables are not garbage collected.
            self._saved_model = model
            self._concrete_fn = model.signatures["serving_default"]

        def call(self, inputs):
            outputs = self._concrete_fn(inputs=tf.convert_to_tensor(inputs))
            return outputs["feature_vector"]

    # Build the classification model architecture
    input_layer = tf.keras.Input(shape=(299, 299, 3))
    features = FeatureExtractor(feature_extractor_model)(input_layer)
    x = layers.Dropout(0.2)(features)
    output_layer = layers.Dense(len(CLASS_NAMES), activation="softmax")(x)
    model = models.Model(inputs=input_layer, outputs=output_layer)

    # Load trained weights
    model.load_weights(os.path.join(os.path.dirname(__file__), "invasive.weights.h5"))
    return model


def preprocess_image(img: Image.Image) -> np.ndarray:
    img = img.resize((299, 299))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    img_array = tf.keras.applications.inception_v3.preprocess_input(img_array)
    return img_array


st.title("Invasive Plant Species Detection")
model = load_model()

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img_array = preprocess_image(image)
    preds = model.predict(img_array)
    pred_idx = np.argmax(preds, axis=1)[0]
    st.write(f"Prediction: {CLASS_NAMES[pred_idx]}")
