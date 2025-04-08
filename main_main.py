# main.py
from flask import Flask, render_template
from image_predict import image_bp
from symptom_predict import symptom_bp

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/uploads"
app.secret_key = "supersecretkey"

# Register Blueprints
app.register_blueprint(image_bp)
app.register_blueprint(symptom_bp)

@app.route("/")
def home():
    return render_template("image_home.html")

if __name__ == "__main__":
    app.run(debug=True)
