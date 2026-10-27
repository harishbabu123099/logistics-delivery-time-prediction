# Week 4 — Predictive Modeling and Optimization in Logistics Systems

## Project objective
Build a reproducible demonstration that predicts shipment delivery time and uses the prediction to propose practical logistics improvements.

## Files
- `logistics_prediction.py` — generates hypothetical data, trains two regression models, evaluates them, and exports CSV outputs.
- `Week_4_Predictive_Logistics_Modeling_Report.docx` — detailed written report and methodology.

## Requirements
Python 3.9+ recommended.
```bash
pip install pandas numpy scikit-learn matplotlib
python logistics_prediction.py
```

## Dataset
The code generates 1,000 simulated shipments with distance, weight, traffic, weather, transport mode, dispatch hour, number of stops, and delivery time. It is synthetic educational data, not real company data.

## Models and metrics
The script compares Linear Regression and Random Forest Regression using a held-out test set and 5-fold cross-validation. It reports MAE, RMSE, and R². Actual results are printed when the code is run; do not claim illustrative figures as measured operational results.

## Suggested next steps
1. Replace simulated data with approved, anonymised historical shipment records.
2. Use a chronological split for real forecasting so future shipments do not leak into training.
3. Compare against a simple baseline (for example, median delivery time by transport mode).
4. Tune model parameters only on training data.
5. Pilot recommendations and track on-time delivery, cost per shipment, and utilisation before scaling.
