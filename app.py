
from pathlib import Path

import pandas as pd
from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

app = Flask(__name__)

# ==========================================
# 1. LOAD DATASET
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "house_prices.csv"

FEATURES = ["size_sqft", "bedrooms", "age_years"]
TARGET = "price_lakhs"


def load_dataset():
    df = pd.read_csv(DATASET_PATH)

    required_columns = FEATURES + [TARGET]

    missing = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required CSV columns: {missing}"
        )

    df = df[required_columns].apply(
        pd.to_numeric, errors="coerce"
    )

    df = df.dropna()

    if len(df) < 5:
        raise ValueError(
            "The dataset must contain at least five valid rows."
        )

    if (
        (df["size_sqft"] <= 0).any()
        or (df["bedrooms"] <= 0).any()
        or (df["age_years"] < 0).any()
        or (df["price_lakhs"] <= 0).any()
    ):
        raise ValueError(
            "The CSV contains invalid house values."
        )

    return df


# ==========================================
# 2. PREPARE FEATURES AND TARGET
# ==========================================

df = load_dataset()

X = df[FEATURES]
y = df[TARGET]


# ==========================================
# 3. SPLIT DATA INTO TRAINING AND TESTING
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ==========================================
# 4. CREATE AND TRAIN DECISION TREE MODEL
# ==========================================

model = DecisionTreeRegressor(
    max_depth=4,
    random_state=42
)

model.fit(X_train, y_train)


# ==========================================
# 5. EVALUATE MODEL
# ==========================================

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)


# ==========================================
# 6. HOME PAGE
# ==========================================

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    house_details = None
    error = None

    if request.method == "POST":
        try:
            size = float(request.form["size"])
            bedrooms = int(request.form["bedrooms"])
            age = float(request.form["age"])

            if (
                size <= 0
                or bedrooms <= 0
                or age < 0
                or not all(map(
                    np.isfinite, [size, bedrooms, age]
                ))
            ):
                raise ValueError(
                    "Enter valid positive house values."
                )

            input_data = pd.DataFrame(
                [[size, bedrooms, age]],
                columns=FEATURES
            )

            prediction = float(
                model.predict(input_data)[0]
            )

            prediction = max(0.0, prediction)

            house_details = {
                "size": size,
                "bedrooms": bedrooms,
                "age": age
            }

        except (ValueError, KeyError):
            error = (
                "Please enter valid house details "
                "and try again."
            )

    metrics = {
        "total_records": len(df),
        "train_records": len(X_train),
        "test_records": len(X_test),
        "mae": round(float(mae), 2),
        "rmse": round(float(rmse), 2),
        "r2": round(float(r2), 3)
    }

    return render_template(
        "index.html",
        prediction=prediction,
        house=house_details,
        error=error,
        metrics=metrics
    )


# ==========================================
# 7. RUN FLASK APPLICATION
# ==========================================

if __name__ == "__main__":
    app.run(debug=True)
