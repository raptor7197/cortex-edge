import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

"""Train & evaluate the learned router (PLAN 15.13).

Trains a RandomForest classifier on datasets/routing_training.csv
(prompt features + device state -> best_route labels) and persists
experiments/router.joblib. Compares against the majority class.

Usage: python scripts/train_router.py
"""

import joblib
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.router.learned_router import FEATURE_ORDER

CSV_PATH = Path("datasets") / "routing_training.csv"
OUT_PATH = Path("experiments") / "router.joblib"


def main():
    if not CSV_PATH.exists():
        raise SystemExit(
            f"{CSV_PATH} missing — run scripts/build_dataset.py first"
        )
    df = pd.read_csv(CSV_PATH).dropna(subset=FEATURE_ORDER + ["best_route"])
    if len(df) < 20:
        print(f"Warning: only {len(df)} labelled rows; small dataset.")
    X, y = df[FEATURE_ORDER], df["best_route"]

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
    except ValueError:
        print("Warning: too few samples per class for stratified split; "
              "using a random split.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    model = Pipeline([
        ("scale", StandardScaler()),
        ("classifier", RandomForestClassifier(
            n_estimators=250, max_depth=12, class_weight="balanced", random_state=42
        )),
    ])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    print("Classification report (test set):")
    print(classification_report(y_test, pred, zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, pred))
    print(f"Baseline (majority class): {y.value_counts().idxmax()} "
          f"({y.value_counts().max() / len(y):.2f})")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, OUT_PATH)
    print(f"\nSaved {OUT_PATH}")


if __name__ == "__main__":
    main()