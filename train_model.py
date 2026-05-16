from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Dataset" / "Table_1.csv"
ARTIFACT_DIR = BASE_DIR / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "employee_attrition_model.joblib"
METRICS_PATH = ARTIFACT_DIR / "employee_attrition_model.metrics.json"

TARGET = "Stay/Left"
DROP_COLUMNS = ["table id", "name", "phone number"]


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df.columns = [column.strip() for column in df.columns]
    df = df.dropna(subset=[TARGET])
    return df


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    numeric_features = features.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = [
        column for column in features.columns if column not in numeric_features
    ]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", make_one_hot_encoder()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )


def evaluate_model(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict:
    predictions = model.predict(x_test)
    return {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(
            y_test, predictions, pos_label="Left", average="binary", zero_division=0
        ),
        "recall": recall_score(
            y_test, predictions, pos_label="Left", average="binary", zero_division=0
        ),
        "f1": f1_score(
            y_test, predictions, pos_label="Left", average="binary", zero_division=0
        ),
    }


def build_options(features: pd.DataFrame) -> dict:
    options = {}
    for column in features.columns:
        if features[column].dtype == "object":
            values = sorted(features[column].dropna().astype(str).unique().tolist())
            options[column] = values
    return options


def build_defaults(features: pd.DataFrame) -> dict:
    defaults = {}
    for column in features.columns:
        if features[column].dtype == "object":
            mode = features[column].mode(dropna=True)
            defaults[column] = "" if mode.empty else str(mode.iloc[0])
        else:
            defaults[column] = float(features[column].median())
    return defaults


def train() -> dict:
    ARTIFACT_DIR.mkdir(exist_ok=True)
    df = load_data()

    features = df.drop(columns=[TARGET, *DROP_COLUMNS], errors="ignore")
    target = df[TARGET].astype(str)

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target,
    )

    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1000, solver="liblinear"),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=42),
        "Naive Bayes": GaussianNB(),
    }

    results = {}
    trained_models = {}
    for name, estimator in candidates.items():
        model = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(features)),
                ("model", estimator),
            ]
        )
        model.fit(x_train, y_train)
        results[name] = evaluate_model(model, x_test, y_test)
        trained_models[name] = model

    best_model_name = max(results, key=lambda item: results[item]["f1"])
    artifact = {
        "project": "Employee Attrition Prediction",
        "model": trained_models[best_model_name],
        "best_model": best_model_name,
        "features": features.columns.tolist(),
        "metrics": results,
        "options": build_options(features),
        "defaults": build_defaults(features),
        "positive_label": "Left",
    }

    joblib.dump(artifact, MODEL_PATH)
    METRICS_PATH.write_text(
        json.dumps(
            {
                "best_model": best_model_name,
                "metrics": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return artifact


if __name__ == "__main__":
    trained_artifact = train()
    best = trained_artifact["best_model"]
    print(f"Saved {best} model to {MODEL_PATH}")
