import os
import tempfile

import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, render_template, request
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "vgg16_brain_tumor_best_glioma_recall.weights.h5")
IMG_SIZE = (224, 224)
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]


def build_model():
    base_model = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation="relu")(x)
    x = Dense(1024, activation="relu")(x)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.2)(x)
    output = Dense(len(CLASS_NAMES), activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=output, name="vgg16_brain_tumor")
    model.load_weights(MODEL_PATH)
    return model


model = build_model()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "classes": CLASS_NAMES})


@app.route("/api/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400

    uploaded_file = request.files["file"]
    if uploaded_file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
        uploaded_file.save(temp_file.name)
        temp_path = temp_file.name

    try:
        image = tf.keras.utils.load_img(temp_path, target_size=IMG_SIZE)
        image_array = tf.keras.utils.img_to_array(image)
        image_batch = np.expand_dims(image_array, axis=0)
        probabilities = model.predict(preprocess_input(image_batch), verbose=0)[0]

        predicted_index = int(np.argmax(probabilities))
        predicted_class = CLASS_NAMES[predicted_index]
        confidence = float(probabilities[predicted_index])

        return jsonify(
            {
                "label": predicted_class,
                "confidence": round(confidence, 4),
                "probabilities": {
                    class_name: round(float(probabilities[idx]), 4)
                    for idx, class_name in enumerate(CLASS_NAMES)
                },
                "summary": f"{predicted_class.title()} is the most likely diagnosis with {confidence * 100:.1f}% confidence.",
            }
        )
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
