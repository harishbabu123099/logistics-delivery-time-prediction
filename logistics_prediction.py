"""
Week 4: Predictive Modeling and Optimization in Logistics Systems
Educational project using a simulated logistics dataset.

Install:
    pip install pandas numpy scikit-learn matplotlib

Run:
    python logistics_prediction.py

Note: This project creates synthetic data for demonstration. Replace it with
approved real shipment records before making operational decisions.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

RANDOM_STATE = 42

def create_sample_data(n=1000, random_state=RANDOM_STATE):
    """Create reproducible hypothetical shipment records."""
    rng = np.random.default_rng(random_state)
    distance = rng.uniform(10, 1000, n)
    weight = rng.uniform(1, 1500, n)
    traffic = rng.choice(["Low", "Medium", "High"], n, p=[0.30, 0.45, 0.25])
    weather = rng.choice(["Clear", "Rain", "Severe"], n, p=[0.70, 0.24, 0.06])
    mode = rng.choice(["Road", "Rail", "Air"], n, p=[0.70, 0.23, 0.07])
    dispatch_hour = rng.integers(0, 24, n)
    stops = rng.integers(0, 6, n)

    traffic_delay = pd.Series(traffic).map({"Low": 0.2, "Medium": 1.5, "High": 4.0}).to_numpy()
    weather_delay = pd.Series(weather).map({"Clear": 0.0, "Rain": 2.0, "Severe": 6.0}).to_numpy()
    mode_factor = pd.Series(mode).map({"Road": 1.0, "Rail": 0.72, "Air": 0.22}).to_numpy()
    peak_delay = np.where((dispatch_hour >= 7) & (dispatch_hour <= 10) |
                          (dispatch_hour >= 16) & (dispatch_hour <= 19), 1.8, 0.0)
    noise = rng.normal(0, 2.5, n)

    # Hypothetical delivery-time target (hours); coefficients are illustrative.
    delivery_hours = (
        1.5 + (distance / 60) * mode_factor
        + (weight / 1500) * 2.0
        + traffic_delay + weather_delay + peak_delay
        + stops * 0.7 + noise
    )
    delivery_hours = np.maximum(delivery_hours, 0.5)

    return pd.DataFrame({
        "distance_km": distance.round(1),
        "weight_kg": weight.round(1),
        "traffic": traffic,
        "weather": weather,
        "transport_mode": mode,
        "dispatch_hour": dispatch_hour,
        "number_of_stops": stops,
        "delivery_hours": delivery_hours.round(2),
    })

def main():
    df = create_sample_data()
    target = "delivery_hours"
    X = df.drop(columns=[target])
    y = df[target]

    numeric_features = ["distance_km", "weight_kg", "dispatch_hour", "number_of_stops"]
    categorical_features = ["traffic", "weather", "transport_mode"]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocess = ColumnTransformer([
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features),
    ])

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, min_samples_leaf=2, random_state=RANDOM_STATE,
            n_jobs=-1
        ),
    }

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )
    results = []
    fitted = {}
    for name, estimator in models.items():
        pipe = Pipeline([("preprocess", preprocess), ("model", estimator)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        mae = mean_absolute_error(y_test, pred)
        r2 = r2_score(y_test, pred)
        cv_mae = -cross_val_score(
            pipe, X_train, y_train, cv=5,
            scoring="neg_mean_absolute_error", n_jobs=None
        ).mean()
        results.append({"Model": name, "MAE_hours": mae,
                        "RMSE_hours": rmse, "R2": r2,
                        "5fold_CV_MAE_hours": cv_mae})
        fitted[name] = pipe

    results_df = pd.DataFrame(results).sort_values("MAE_hours")
    print("\\nMODEL EVALUATION (synthetic data; results are illustrative)")
    print(results_df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\\nSample rows:")
    print(df.head().to_string(index=False))
    df.to_csv("simulated_logistics_shipments.csv", index=False)
    results_df.to_csv("model_evaluation_results.csv", index=False)
    print("\\nSaved simulated_logistics_shipments.csv and model_evaluation_results.csv")

if __name__ == "__main__":
    main()
