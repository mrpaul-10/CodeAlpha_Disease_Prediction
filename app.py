
import json
from pathlib import Path

import joblib
import pandas as pd
import xgboost as xgb
from flask import Flask, render_template, request

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent


# Find model files
def resolve_asset(filename):
    candidates = [
        BASE_DIR / filename,
        BASE_DIR / "notebook" / filename,
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Could not find {filename}"
    )


# Load trained model and configuration
def load_model():

    model_path = resolve_asset(
        "diabetes_xgb_model.joblib"
    )

    config_path = resolve_asset(
        "model_config.json"
    )

    model = joblib.load(model_path)

    with open(config_path, encoding="utf-8") as file:
        config = json.load(file)

    return model, config


model, config = load_model()

threshold = config["threshold"]
features = config["features"]
medians = config["medians"]


# Example patients
EXAMPLES = {

    "high": {
        "preg": 6,
        "age": 50,
        "dpf": 0.63,
        "glucose": 165,
        "bmi": 34.0,
        "bp": 72,
        "skin": 35,
        "insulin": 0,
        "u_glu": False,
        "u_bmi": False,
        "u_bp": False,
        "u_skin": False,
        "u_ins": True,
    },

    "low": {
        "preg": 1,
        "age": 24,
        "dpf": 0.25,
        "glucose": 92,
        "bmi": 23.0,
        "bp": 68,
        "skin": 22,
        "insulin": 70,
        "u_glu": False,
        "u_bmi": False,
        "u_bp": False,
        "u_skin": False,
        "u_ins": False,
    }

}


DEFAULTS = {
    "preg": 2,
    "age": 35,
    "dpf": 0.45,
    "glucose": 120,
    "bmi": 28.0,
    "bp": 70,
    "skin": 25,
    "insulin": 80,
    "u_glu": False,
    "u_bmi": False,
    "u_bp": False,
    "u_skin": False,
    "u_ins": True,
}


# Read patient input
def get_form_values(form):

    values = {}

    for key, default in DEFAULTS.items():

        if isinstance(default, bool):
            values[key] = form.get(key) == "on"

        elif isinstance(default, int):
            values[key] = int(form.get(key, default))

        else:
            values[key] = float(form.get(key, default))

    return values


# Make prediction
def make_prediction(values):

    row = {

        "Pregnancies": values["preg"],

        "Age": values["age"],

        "DiabetesPedigreeFunction": values["dpf"],

        "Glucose": (
            medians["Glucose"]
            if values["u_glu"]
            else values["glucose"]
        ),

        "BMI": (
            medians["BMI"]
            if values["u_bmi"]
            else values["bmi"]
        ),

        "BloodPressure": (
            medians["BloodPressure"]
            if values["u_bp"]
            else values["bp"]
        ),

        "SkinThickness": (
            medians["SkinThickness"]
            if values["u_skin"]
            else values["skin"]
        ),

        "Insulin": (
            medians["Insulin"]
            if values["u_ins"]
            else values["insulin"]
        ),

    }

    X_new = pd.DataFrame([row])[features]

    # Calculate probability
    risk = float(
        model.predict_proba(X_new)[0, 1]
    )

    high = risk >= threshold

    # Individual feature contributions
    contrib = model.get_booster().predict(
        xgb.DMatrix(X_new),
        pred_contribs=True
    )[0][:-1]

    contributions = sorted(
        [
            {
                "name": name,
                "value": float(value)
            }
            for name, value in zip(features, contrib)
        ],
        key=lambda x: x["value"]
    )

    # Overall feature importance
    importance = sorted(
        [
            {
                "name": name,
                "value": float(value)
            }
            for name, value in zip(
                features,
                model.feature_importances_
            )
        ],
        key=lambda x: x["value"],
        reverse=True
    )

    # Unknown values
    unknown_used = []

    unknown_mapping = [
        ("Glucose", "u_glu"),
        ("BMI", "u_bmi"),
        ("Blood Pressure", "u_bp"),
        ("Skin Thickness", "u_skin"),
        ("Insulin", "u_ins")
    ]

    for name, key in unknown_mapping:

        if values[key]:
            unknown_used.append(name)

    return {
        "risk": risk,
        "high": high,
        "contributions": contributions,
        "importance": importance,
        "unknown_used": unknown_used
    }


# Home page
@app.route("/", methods=["GET", "POST"])
def index():

    values = DEFAULTS.copy()

    result = None
    error = None

    if request.method == "POST":

        try:

            values = get_form_values(request.form)

            result = make_prediction(values)

        except (ValueError, TypeError) as exc:

            error = f"Please check your input values: {exc}"

    return render_template(
        "index.html",
        values=values,
        result=result,
        error=error,
        threshold=threshold
    )


# Load example patient
@app.route("/example/<name>", methods=["POST"])
def load_example(name):

    if name not in EXAMPLES:
        return "Example not found", 404

    values = EXAMPLES[name]

    result = make_prediction(values)

    return render_template(
        "index.html",
        values=values,
        result=result,
        error=None,
        threshold=threshold
    )


if __name__ == "__main__":

    app.run(debug=True)