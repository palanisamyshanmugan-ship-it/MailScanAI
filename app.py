from flask import Flask, render_template, request, jsonify
import os
import cv2
import numpy as np
import pytesseract
import pickle
import shutil
from werkzeug.utils import secure_filename

# ==========================================
# TESSERACT CONFIGURATION
# ==========================================

if os.name == "nt":
    pytesseract.pytesseract.tesseract_cmd = (
        r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    )
else:
    tesseract_path = shutil.which("tesseract")
    if tesseract_path:
        pytesseract.pytesseract.tesseract_cmd = tesseract_path

# ==========================================
# FLASK APP
# ==========================================

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
MODEL_FOLDER = "model"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {
    "png",
    "jpg",
    "jpeg",
    "bmp",
    "webp"
}

MODEL_PATH = os.path.join(
    MODEL_FOLDER,
    "email_classifier.pkl"
)

ENCODER_PATH = os.path.join(
    MODEL_FOLDER,
    "label_encoder.pkl"
)

classifier = None
vectorizer = None
label_encoder = None

# ==========================================
# LOAD MODEL
# ==========================================

try:

    with open(MODEL_PATH, "rb") as f:
        saved_data = pickle.load(f)

    classifier = saved_data["model"]
    vectorizer = saved_data["vectorizer"]

    with open(ENCODER_PATH, "rb") as f:
        label_encoder = pickle.load(f)

    print("Model loaded successfully")

except Exception as e:

    print("Model loading error:", str(e))

# ==========================================
# FILE VALIDATION
# ==========================================

def allowed_file(filename):

    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )

# ==========================================
# OCR FUNCTION
# ==========================================

def extract_text(image_path):

    image = cv2.imread(image_path)

    if image is None:
        raise Exception(
            "Unable to read image."
        )

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    denoise = cv2.fastNlMeansDenoising(
        gray
    )

    thresh = cv2.adaptiveThreshold(
        denoise,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )

    text = pytesseract.image_to_string(
        thresh
    )

    return text.strip()

# ==========================================
# HOME PAGE
# ==========================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )

# ==========================================
# CLASSIFICATION API
# ==========================================

@app.route(
    "/classify",
    methods=["POST"]
)
def classify():

    try:

        if classifier is None:

            return jsonify({
                "error":
                "Model not loaded"
            }), 500

        if (
            "image"
            not in request.files
        ):

            return jsonify({
                "error":
                "No image uploaded"
            }), 400

        file = request.files["image"]

        if file.filename == "":

            return jsonify({
                "error":
                "No file selected"
            }), 400

        if not allowed_file(
            file.filename
        ):

            return jsonify({
                "error":
                "Invalid file type"
            }), 400

        filename = secure_filename(
            file.filename
        )

        filepath = os.path.join(
            UPLOAD_FOLDER,
            filename
        )

        file.save(filepath)

        extracted_text = extract_text(
            filepath
        )

        if (
            len(
                extracted_text.strip()
            ) < 5
        ):

            return jsonify({
                "error":
                "No readable text found"
            }), 400

        text_vector = (
            vectorizer.transform(
                [extracted_text]
            )
        )

        prediction = (
            classifier.predict(
                text_vector
            )[0]
        )

        probabilities = (
            classifier.predict_proba(
                text_vector
            )[0]
        )

        label = (
            label_encoder
            .inverse_transform(
                [prediction]
            )[0]
        )

        confidence = round(
            float(
                np.max(
                    probabilities
                )
            ) * 100,
            2
        )

        classes = (
            label_encoder
            .inverse_transform(
                np.arange(
                    len(probabilities)
                )
            )
        )

        all_scores = {}

        for cls, score in zip(
            classes,
            probabilities
        ):

            all_scores[
                cls
            ] = round(
                float(score) * 100,
                2
            )

        return jsonify({

            "label": label,

            "confidence":
            confidence,

            "all_scores":
            all_scores,

            "extracted_text":
            extracted_text

        })

    except Exception as e:

        return jsonify({
            "error":
            str(e)
        }), 500

# ==========================================
# RUN APP
# ==========================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=True
    )
