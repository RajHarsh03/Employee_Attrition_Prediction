from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from django.shortcuts import render


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "artifacts" / "employee_attrition_model.joblib"


def load_artifact() -> dict:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model artifact is missing. Run `python train_model.py` in Employee_Attrition."
        )
    return joblib.load(MODEL_PATH)


def coerce_value(value: str, default):
    if isinstance(default, float):
        try:
            return float(value)
        except (TypeError, ValueError):
            return default
    return value or default


def index(request):
    artifact = load_artifact()
    defaults = artifact["defaults"]
    form_values = defaults.copy()
    prediction = None

    if request.method == "POST":
        for feature in artifact["features"]:
            form_values[feature] = coerce_value(request.POST.get(feature, ""), defaults[feature])

        input_frame = pd.DataFrame([form_values], columns=artifact["features"])
        model = artifact["model"]
        predicted_label = model.predict(input_frame)[0]

        leave_probability = None
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_frame)[0]
            classes = list(model.classes_)
            if artifact["positive_label"] in classes:
                leave_probability = float(probabilities[classes.index(artifact["positive_label"])])

        prediction = {
            "label": predicted_label,
            "status": "Likely to Leave" if predicted_label == artifact["positive_label"] else "Likely to Stay",
            "leave_probability": None
            if leave_probability is None
            else round(leave_probability * 100, 2),
        }

    return render(
        request,
        "predictor/index.html",
        {
            "features": artifact["features"],
            "options": artifact["options"],
            "defaults": defaults,
            "values": form_values,
            "prediction": prediction,
            "best_model": artifact["best_model"],
            "metrics": artifact["metrics"][artifact["best_model"]],
        },
    )
