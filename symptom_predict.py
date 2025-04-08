from flask import Blueprint, render_template, request, session
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np

symptom_bp = Blueprint('symptom_bp', __name__, template_folder='templates')

# Load the dataset
data = pd.read_csv("cattle-Disease123.csv")
data.rename(columns={"Discease": "Disease"}, inplace=True)
data = data.dropna(subset=["Disease"])

# Convert categorical features to numerical
X = pd.get_dummies(data.drop(columns=["Disease"]), drop_first=False)

# Encode target variable
y = data["Disease"]
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X, y_encoded)

# Store symptoms
symptoms = list(X.columns)
importances = model.feature_importances_
sorted_indices = np.argsort(importances)[::-1]  # Symptoms sorted by importance

@symptom_bp.route("/symptom")
def symptom_home():
    """Initialize session and start symptom questioning process."""
    session["remaining_indices"] = list(range(len(X)))  # Store remaining cases
    session["asked_symptoms"] = []  # Store asked symptom names
    return next_question()

@symptom_bp.route("/next", methods=["POST"])
def next_question():
    """Ask the next symptom question and filter if 'Yes' is clicked."""
    remaining_indices = session.get("remaining_indices", [])
    asked_symptoms = session.get("asked_symptoms", [])

    # Process the user's answer
    if "answer" in request.form and session.get("last_symptom"):
        last_symptom = session["last_symptom"]
        if request.form["answer"] == "yes":
            # **Filter the dataset based on "Yes" response**
            remaining_indices = [idx for idx in remaining_indices if X.loc[idx, last_symptom] == 1]
            session["remaining_indices"] = remaining_indices

    # **Check if we reached a conclusion**
    if len(remaining_indices) == 1:
        disease = label_encoder.inverse_transform([y_encoded[remaining_indices[0]]])[0]
        return render_template("symptom_predict.html", result=f"The likely disease is: {disease}")

    if len(remaining_indices) == 0:
        return render_template("symptom_predict.html", result="No matching diseases found.")

    # **Ask the next most important question**
    for idx in sorted_indices:
        if symptoms[idx] not in asked_symptoms and X.iloc[remaining_indices, idx].sum() > 0:
            session["last_symptom"] = symptoms[idx]
            asked_symptoms.append(symptoms[idx])
            session["asked_symptoms"] = asked_symptoms
            return render_template("symptom_predict.html", question=f"Do you observe {symptoms[idx]}?", symptom=symptoms[idx])

    return render_template("symptom_predict.html", result="Multiple possible diseases detected.")
