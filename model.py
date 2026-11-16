
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
DATASET_GLOB = "thoracic_uci_data*.csv"
MODEL_PATH = ROOT / "thoracic_model.pkl"

TARGET_COLUMN = "Risk1Y"
REQUIRED_FEATURES = [
    "DGN",
    "PRE4",
    "PRE5",
    "PRE6",
    "PRE7",
    "PRE8",
    "PRE9",
    "PRE10",
    "PRE11",
    "PRE14",
    "PRE17",
    "PRE19",
    "PRE25",
    "PRE30",
    "PRE32",
    "AGE",
]


def find_dataset() -> Path:
    matches = sorted(ROOT.glob(DATASET_GLOB))
    if not matches:
        raise FileNotFoundError(
            f"No thoracic dataset found in {ROOT}. Expected a file matching '{DATASET_GLOB}'."
        )
    return matches[0]


def load_dataset() -> pd.DataFrame:
    path = find_dataset()
    df = pd.read_csv(path)
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Dataset '{path.name}' is missing the required target column '{TARGET_COLUMN}'.")
    return df


def prepare_features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    data = df.copy()
    data[TARGET_COLUMN] = data[TARGET_COLUMN].astype(str).str.strip().str.upper()
    data = data.dropna(subset=[TARGET_COLUMN])

    missing = [col for col in REQUIRED_FEATURES if col not in data.columns]
    if missing:
        raise ValueError(f"Dataset is missing required feature columns: {missing}")

    X = data[REQUIRED_FEATURES]
    y = data[TARGET_COLUMN]
    return X, y


def build_pipeline() -> Pipeline:
    numeric_columns = ["PRE4", "PRE5", "AGE"]
    categorical_columns = [col for col in REQUIRED_FEATURES if col not in numeric_columns]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_columns,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_columns,
            ),
        ]
    )

    model = LogisticRegression(
        max_iter=5000,
        class_weight="balanced",
        random_state=42,
        solver="liblinear",
    )

    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def train_model() -> tuple[Pipeline, dict[str, float]]:
    df = load_dataset()
    X, y = prepare_features_and_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    probability = pipeline.predict_proba(X_test)[:, 1]
    positive_label = "T"
    y_test_binary = (y_test == positive_label).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "f1_score": f1_score(y_test, predictions, pos_label="T"),
        "roc_auc": roc_auc_score(y_test_binary, probability),
        "classification_report": classification_report(y_test, predictions, digits=4),
    }

    save_model(pipeline)
    return pipeline, metrics


def save_model(model: Pipeline, path: Path | None = None) -> None:
    destination = path or MODEL_PATH
    with destination.open("wb") as file:
        pickle.dump(model, file)


def load_model(path: Path | None = None) -> Pipeline:
    destination = path or MODEL_PATH
    with destination.open("rb") as file:
        return pickle.load(file)


def predict_risk(data: dict[str, Any] | pd.DataFrame | Iterable[dict[str, Any]]) -> Any:
    model = load_model()

    if isinstance(data, dict):
        rows = [data]
    elif isinstance(data, pd.DataFrame):
        rows = data.copy()
    else:
        rows = list(data)

    frame = pd.DataFrame(rows)
    frame = frame.reindex(columns=REQUIRED_FEATURES, fill_value=None)
    prediction = model.predict(frame)
    return prediction[0] if len(prediction) == 1 else prediction


if __name__ == "__main__":
    trained_model, metrics = train_model()
    print("Model training complete.")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1-score (T): {metrics['f1_score']:.4f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print("\nClassification report:\n")
    print(metrics["classification_report"])
    print(f"Saved model to: {MODEL_PATH}")