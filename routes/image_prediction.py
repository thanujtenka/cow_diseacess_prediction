from flask import Blueprint, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import UnidentifiedImageError
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

bp = Blueprint("image_prediction", __name__)
UPLOAD_FOLDER = "static/uploads"

# Load the model
try:
    model = load_model('cattle_da_cow.h5')
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    model = None

# Define known categories from the model
model_classes = {
    'Bovine': 0, 'Dermatitis': 1, 'Ecthym': 2, 'Foot-and-mouth disease': 3,
    'a bacterial infection of the eyes in cattle': 4, 'abscesses': 5, 'healthy': 6,
    'lumpy skin': 7, 'photosensitization': 8, 'ringworm': 9
}
sorted_class_names = [k for k, v in sorted(model_classes.items(), key=lambda item: item[1])]
recognized_categories = set(sorted_class_names)

def predict_image(img_path, confidence_threshold=0.2):
    try:
        if not os.path.exists(img_path):
            logging.error("File not found.")
            return "Mismatch: File not found."

        # Load and preprocess the image
        img = image.load_img(img_path, target_size=(128, 128))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Normalize

        # Make a prediction
        prediction = model.predict(img_array)
        if prediction.shape != (1, len(sorted_class_names)):
            logging.error("Unexpected model output shape.")
            return "Mismatch: Unexpected model output shape."

        prediction = prediction[0]
        highest_confidence_idx = np.argmax(prediction)
        highest_confidence_value = prediction[highest_confidence_idx]
        predicted_class = sorted_class_names[highest_confidence_idx]

        logging.info(f"Predicted Class: {predicted_class}, Confidence: {highest_confidence_value:.2f}")

        if predicted_class not in recognized_categories:
            logging.warning("Mismatch: The image does not belong to the trained categories.")
            return "Mismatch: The image does not belong to the trained categories."

        matched_classes = [(sorted_class_names[idx], confidence) for idx, confidence in enumerate(prediction) if confidence > confidence_threshold]

        if matched_classes:
            return matched_classes
        else:
            return "Mismatch: No class matched above the confidence threshold."

    except UnidentifiedImageError:
        logging.error("The selected file is not a valid image.")
        return "Mismatch: The selected file is not a valid image."
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        return f"Mismatch: {str(e)}"

@bp.route("/image_upload", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if "file" not in request.files:
            return render_template("index.html", error="No file part")

        file = request.files["file"]
        if file.filename == "":
            return render_template("index.html", error="No selected file")

        if file:
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)
            
            result = predict_image(file_path)

            if isinstance(result, list):
                return render_template("index.html", result=result, image_url=file_path)
            else:
                return render_template("index.html", error=result, image_url=file_path)

    return render_template("index.html")
