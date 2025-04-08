from flask import Blueprint, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from PIL import UnidentifiedImageError, Image
import os
import logging

# Flask Blueprint setup
image_bp = Blueprint('image_bp', __name__, template_folder='templates')
UPLOAD_FOLDER = "static/uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load disease model
try:
    disease_model = load_model('cattle_da_cow.h5')
    logging.info("✅ Disease model loaded successfully.")
except Exception as e:
    logging.error(f"❌ Error loading disease model: {e}")
    disease_model = None

# Load MobileNetV2 for cow detection
try:
    cow_detector = MobileNetV2(weights='imagenet')
    logging.info("✅ Cow detection model loaded successfully.")
except Exception as e:
    logging.error(f"❌ Error loading cow detection model: {e}")
    cow_detector = None

# Disease model classes
model_classes = {
    'Bovine': 0, 'Dermatitis': 1, 'Ecthym': 2, 'Foot-and-mouth disease': 3,
    'a bacterial infection of the eyes in cattle': 4, 'abscesses': 5, 'healthy': 6,
    'lumpy skin': 7, 'photosensitization': 8, 'ringworm': 9
}
sorted_class_names = [k for k, v in sorted(model_classes.items(), key=lambda item: item[1])]
recognized_categories = set(sorted_class_names)

# Preprocess image for MobileNetV2 (cow detection)
def preprocess_for_mobilenet(image):
    img = image.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# Check if the image is a cow
def is_cow(img):
    try:
        mobilenet_input = preprocess_for_mobilenet(img)
        predictions = cow_detector.predict(mobilenet_input)
        decoded_predictions = decode_predictions(predictions, top=3)[0]

        logging.info(f"🔍 Cow detection predictions: {decoded_predictions}")

        # Look for cow-related labels
        cow_found = any(label in ['cow', 'ox', 'bull', 'calf'] for _, label, _ in decoded_predictions)
        if cow_found:
            logging.info("✅ Cow detected.")
            return True
        else:
            logging.warning("🚫 Not a cow.")
            return False

    except Exception as e:
        logging.error(f"❌ Error during cow detection: {e}")
        return False

# Preprocess image for disease model
def preprocess_for_disease_model(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

# Predict disease in the image
def predict_image(img_path):
    try:
        if not os.path.exists(img_path):
            logging.error("❌ File not found.")
            return "Mismatch: File not found."

        img_array = preprocess_for_disease_model(img_path)

        # Predict using the disease model
        prediction = disease_model.predict(img_array)

        if prediction.shape != (1, len(sorted_class_names)):
            logging.error("❌ Unexpected model output shape.")
            return "Mismatch: Unexpected model output shape."

        highest_confidence_idx = np.argmax(prediction)
        predicted_class = sorted_class_names[highest_confidence_idx]

        logging.info(f"🎯 Predicted Class: {predicted_class}")

        if predicted_class not in recognized_categories:
            logging.warning("🚫 The image does not match the trained categories.")
            return "Mismatch: The image does not belong to the trained categories."

        return predicted_class

    except UnidentifiedImageError:
        logging.error("❌ Invalid image format.")
        return "Mismatch: The selected file is not a valid image."
    except Exception as e:
        logging.error(f"❌ Error during prediction: {e}")
        return f"Mismatch: {str(e)}"

# Flask Route for Image Upload and Prediction
@image_bp.route("/image", methods=["GET", "POST"])
def upload_and_predict():
    if request.method == "POST":
        if "file" not in request.files:
            logging.error("❌ No file part.")
            return render_template("image_predict.html", error="❌ No file part")

        file = request.files["file"]
        if file.filename == "":
            logging.error("❌ No file selected.")
            return render_template("image_predict.html", error="❌ No selected file")

        if file:
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            try:
                file.save(file_path)
                img = Image.open(file_path)

                # Step 1: Cow detection
                if not is_cow(img):
                    logging.warning("🚫 Not a cow detected.")
                    return render_template(
                        "image_predict.html",
                        error="🚫 Not a cow detected.",
                        image_url=file_path,
                    )

                # Step 2: Disease Prediction
                result = predict_image(file_path)

                if isinstance(result, str):
                    logging.error(f"🚫 Prediction Error: {result}")
                    return render_template(
                        "image_predict.html",
                        error=result,
                        image_url=file_path,
                    )
                else:
                    return render_template(
                        "image_predict.html",
                        result=f"✅ Detected disease: {result}",
                        image_url=file_path,
                    )

            except UnidentifiedImageError:
                logging.error("❌ Invalid image format.")
                return render_template(
                    "image_predict.html",
                    error="❌ Invalid image format",
                    image_url=None,
                )
            except Exception as e:
                logging.error(f"❌ Unexpected error: {e}")
                return render_template(
                    "image_predict.html",
                    error=f"❌ Error: {e}",
                    image_url=None,
                )

    return render_template("image_predict.html")
