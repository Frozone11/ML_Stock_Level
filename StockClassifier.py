import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# ======================================================
# EDIT THIS IF NEEDED:
EXCEL_PATH = "data/ultrasound_bottom.xlsx"  # or "data/ultrasound_bottom.xlsx"
SHEET_NAME = "Data"  # or "Data" / "Sheet1" if you want by name
# ======================================================


def main():
    print("Loading Excel file...")
    try:
        df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME)
    except Exception as e:
        print(f"ERROR reading Excel file: {e}")
        return

    # Expect columns: Reading, Part
    expected_cols = {"Reading", "Part"}
    if not expected_cols.issubset(df.columns):
        print("ERROR: Expected columns 'Reading' and 'Part' in the Excel file.")
        print("Found columns:", list(df.columns))
        return

    # Clean / ensure numeric
    df = df.copy()
    df["Reading"] = pd.to_numeric(df["Reading"], errors="coerce")
    df["Part"] = pd.to_numeric(df["Part"], errors="coerce")
    df = df.dropna(subset=["Reading", "Part"])

    if df.empty:
        print("ERROR: No usable data after cleaning.")
        return

    print("Data preview:")
    print(df.head())
    print("\nUnique parts:", np.unique(df["Part"]))

    # Features and labels
    X = df[["Reading"]].values        # input: 1D reading
    y = df["Part"].values.astype(int) # output: class label 0–20

    # Train/test split (stratify to keep class balance)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\nTraining RandomForestClassifier...")
    clf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)

    # Predictions and metrics
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print("\n===== CLASSIFICATION RESULTS =====")
    print(f"Accuracy (test): {acc:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=sorted(np.unique(y)))
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=sorted(np.unique(y)),
        yticklabels=sorted(np.unique(y)),
    )
    plt.xlabel("Predicted Part")
    plt.ylabel("True Part")
    plt.title("Confusion Matrix – RandomForestClassifier")
    plt.tight_layout()
    plt.show()

    # Example predictions across reading range
    print("\nExample predictions:")
    for reading in np.linspace(df["Reading"].min(), df["Reading"].max(), 5):
        pred = clf.predict([[reading]])[0]
        print(f"Reading {reading:.1f} → predicted Part = {pred}")

    print("\nDone!")


if __name__ == "__main__":
    main()
