import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, ConfusionMatrixDisplay


# ======================================================
EXCEL_PATH = "data/ultrasound_bottompart.xlsx"  # file name
SHEET_NAME = "Data"    
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
    if not expected_cols.issubset(set(df.columns)):
        print("ERROR: Expected columns 'Reading' and 'Part' in the Excel file.")
        print("Found columns:", list(df.columns))
        return

    # Drop rows with missing values and convert to numeric
    df = df.copy()
    df["Reading"] = pd.to_numeric(df["Reading"], errors="coerce")
    df["Part"] = pd.to_numeric(df["Part"], errors="coerce")
    df = df.dropna(subset=["Reading", "Part"])

    if df.empty:
        print("ERROR: No usable data after cleaning.")
        return

    print("Data preview:")
    print(df.head())

    # X = input (distance/reading), y = output (covers/part)
    X = df[["Reading"]].values
    y = df["Part"].values

    print("\nUnique Part values:", np.unique(y))

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("\nTraining linear regression model (Part as function of Reading)...")
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate
    y_pred_test = model.predict(X_test)
    r2 = r2_score(y_test, y_pred_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

    print("\n===== MODEL RESULTS =====")
    print(f"Equation: Part ≈ {model.coef_[0]:.6f} * Reading + {model.intercept_:.6f}")
    print(f"R² (test):  {r2:.4f}")
    print(f"RMSE (test): {rmse:.4f} parts")

    # Convert predictions and actual values to integers (classification-like)
    y_test_rounded = np.round(y_test).astype(int)
    y_pred_rounded = np.round(y_pred_test).astype(int)

    # Compute confusion matrix
    cm = confusion_matrix(y_test_rounded, y_pred_rounded)

    # Plot confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()

    # Example predictions
    print("\nExample predictions:")
    for reading in [df["Reading"].min(), df["Reading"].median(), df["Reading"].max()]:
        pred = model.predict([[reading]])[0]
        print(f"Reading {reading:.1f} → predicted Part ≈ {pred:.2f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
