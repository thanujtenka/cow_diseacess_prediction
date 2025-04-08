from flask import Flask, request
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
import numpy as np
from PIL import Image

app = Flask(__name__)

# Load cow disease model
disease_model = load_model('cacow.h5')

# Load MobileNetV2 for cow detection
cow_detector = MobileNetV2(weights='imagenet')

# Preprocess image for disease model
def preprocess_image(image):
    img = image.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Preprocess image for MobileNetV2
def preprocess_for_mobilenet(image):
    img = image.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'

        file = request.files['file']
        if file.filename == '':
            return 'No selected file'

        if file:
            try:
                img = Image.open(file)
                
                # Step 1: Cow detection using MobileNetV2
                mobilenet_input = preprocess_for_mobilenet(img)
                predictions = cow_detector.predict(mobilenet_input)
                decoded_predictions = decode_predictions(predictions, top=3)[0]
                
                # Cow-related terms
                cow_keywords = ["cow", "cattle", "ox", "bull", "calf"]
                cow_found = False
                confidence_score = 0

                for _, label, score in decoded_predictions:
                    if any(keyword in label.lower() for keyword in cow_keywords):
                        cow_found = True
                        confidence_score = score
                        break

                if not cow_found:
                    return "<h1>Not a cow</h1>"

                # Step 2: Check if it's a full cow or part of a cow
                if confidence_score < 0.5:
                    return "<h1>Part of a cow detected — Not a full cow</h1>"

                # Step 3: If full cow → Run disease prediction
                img_array = preprocess_image(img)
                prediction = disease_model.predict(img_array)

                # Assuming class 0 = cow disease
                predicted_class = np.argmax(prediction)
                
                # If the result is not part of the known dataset
                if predicted_class not in [0, 1]:  # Assuming dataset classes are 0 and 1
                    return "<h1>Not a cow</h1>"

                if predicted_class == 0:
                    result = "It's a cow!"
                else:
                    result = "Cow disease detected"

                return f"<h1>{result}</h1>"

            except Exception as e:
                return f"Error: {e}"

    return '''
    <!doctype html>
    <title>Upload Image</title>
    <h1>Upload an image to check if it's a cow</h1>
    <form method=post enctype=multipart/form-data>
        <input type=file name=file>
        <input type=submit value=Upload>
    </form>
    '''

if __name__ == '__main__':
    app.run(debug=True)
