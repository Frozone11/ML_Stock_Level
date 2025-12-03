import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# ============================================================
# EDIT THIS PATH ONLY:
EXCEL_PATH = "data/bottompart_data.xlsx"
SHEET_NAME = "Data"   # Change if needed
# ============================================================


def reshape_wide_to_long(df):
    """
    Converts:
        Part 0 | Part 1 | ... | Part 20
    into a long table:
        covers | distance
    """
    long_list = []

    for col in df.columns:
        col_lower = col.lower().strip()

        if col_lower.startswith("part"):
            try:
                covers = int(col_lower.replace("part", "").strip())
            except ValueError:
                continue

            distances = pd.to_numeric(df[col], errors="coerce").dropna()

            for d in distances:
                long_list.append({"covers": covers, "distance": d})

    return pd.DataFrame(long_list)


def main():
    print("Loading Excel file...")
    df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME)

    print("Reshaping data...")
    data = reshape_wide_to_long(df)

    if data.empty:
        print("ERROR: No usable data found. Check Excel path or sheet name.")
        return

    print("Data preview:")
    print(data.head())

    # NOTE: now we predict covers FROM distance
    X = data[["distance"]].values   # input = distance
    y = data["covers"].values       # output = covers

    print("\nTraining linear regression model...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred_test = model.predict(X_test)

    print("\n===== MODEL RESULTS =====")
    print(f"Equation: covers ≈ {model.coef_[0]:.4f} * distance + {model.intercept_:.4f}")
    print(f"R² (test): {r2_score(y_test, y_pred_test):.4f}")

    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    print(f"RMSE (test): {rmse_test:.4f}")

    # Plot: distance on x-axis, covers on y-axis
    print("\nPlotting regression fit...")
    plt.scatter(X, y, alpha=0.7, label="Data")
    x_line = np.linspace(data["distance"].min(), data["distance"].max(), 200).reshape(-1, 1)
    plt.plot(x_line, model.predict(x_line), color="red", label="Fit", linewidth=2)
    plt.xlabel("Distance")
    plt.ylabel("Covers")
    plt.title("Covers vs Distance – Linear Regression")
    plt.grid(True)
    plt.legend()
    plt.show()

    print("\nDone!")

    print("Loading Excel file...")
    df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME)

    print("Reshaping data...")
    data = reshape_wide_to_long(df)

    if data.empty:
        print("ERROR: No usable data found. Check Excel path or sheet name.")
        return

    print("Data preview:")
    print(data.head())

    X = data[["distance"]].values   # input: distance
    y = data["covers"].values       # target: covers


    print("\nTraining linear regression model...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred_test = model.predict(X_test)

    print("\n===== MODEL RESULTS =====")
    print(f"Equation: distance ≈ {model.coef_[0]:.4f} * covers + {model.intercept_:.4f}")
    print(f"R² (test): {r2_score(y_test, y_pred_test):.4f}")
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    print(f"RMSE (test): {rmse_test:.4f}")


    # Plot
    print("\nPlotting regression fit...")
    plt.scatter(X, y, alpha=0.7, label="Data")
    x_line = np.linspace(0, 20, 200).reshape(-1, 1)
    plt.plot(x_line, model.predict(x_line), color="red", label="Fit", linewidth=2)
    plt.xlabel("Covers")
    plt.ylabel("Distance")
    plt.title("Distance vs Covers – Linear Regression")
    plt.grid(True)
    plt.legend()
    plt.show()

    print("\nDone!")


if __name__ == "__main__":
    main()
