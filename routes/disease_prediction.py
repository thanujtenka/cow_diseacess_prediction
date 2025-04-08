from flask import Blueprint, render_template, request, session
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np

bp = Blueprint("disease_prediction", __name__)
session_data = {}

# Load the dataset
data = pd.read_csv("cattle-Disease123.csv")
data.rename(columns={"Discease": "Disease"}, inplace=True)
data = data.dropna(subset=["Disease"])

X = pd.get_dummies(data.drop(columns=["Disease"]), drop_first=True)
y = data["Disease"]
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

model = RandomForestClassifier(random_state=42)
model.fit(X, y_encoded)

symptoms = list(X.columns)
importances = model.feature_importances_
sorted_indices = np.argsort(importances)[::-1]

@bp.route("/disease_check")
def home():
    session["remaining_indices"] = list(range(len(X)))
    session["asked_symptoms"] = []
    return render_template("result.html", question="Click Start to begin symptom check.")

@bp.route("/next", methods=["POST"])
def next_question():
    asked_symptoms = session.get("asked_symptoms", [])
    remaining_indices = session.get("remaining_indices", [])

    if "answer" in request.form:
        last_symptom = session.get("last_symptom", None)
        if last_symptom is not None:
            answer = request.form["answer"]
            symptom_value = 1 if answer == "yes" else 0
            remaining_indices = [idx for idx in remaining_indices if X.iloc[idx, symptoms.index(last_symptom)] == symptom_value]
            session["remaining_indices"] = remaining_indices

    if len(remaining_indices) == 1:
        disease = label_encoder.inverse_transform([y_encoded[remaining_indices[0]]])[0]
        return render_template("result.html", result=f"The likely disease is: {disease}")

    if len(remaining_indices) == 0:
        return render_template("result.html", result="No matching diseases found.")

    for idx in sorted_indices:
        if symptoms[idx] not in asked_symptoms and X.iloc[remaining_indices, idx].sum() > 0:
            session["last_symptom"] = symptoms[idx]
            asked_symptoms.append(symptoms[idx])
            session["asked_symptoms"] = asked_symptoms
            return render_template("result.html", question=f"Do you observe {symptoms[idx]}?", symptom=symptoms[idx])

    return render_template("result.html", result="Multiple possible diseases detected.")
