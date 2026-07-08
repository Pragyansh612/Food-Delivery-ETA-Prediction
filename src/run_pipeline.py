"""End-to-end pipeline: load -> clean -> engineer features -> split -> train
all three models -> print the final comparison table. This is the single
script referenced in the README for reproducing the headline numbers
without opening the notebook.
"""

import sys
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from clean import clean
from data_loader import load_raw
from evaluate import report_metrics
from features import add_features
from split import time_based_split
from train_models import train_random_forest, train_xgboost


def main():
    df = add_features(clean(load_raw()))
    X_train, X_test, y_train, y_test = time_based_split(df)
    print(f"train: {X_train.shape}, test: {X_test.shape}\n")

    scaler = StandardScaler()
    lin_reg = LinearRegression().fit(scaler.fit_transform(X_train), y_train)
    lin_metrics = report_metrics(y_test, lin_reg.predict(scaler.transform(X_test)), "Linear Regression")

    rf_model = train_random_forest(X_train, y_train)
    rf_metrics = report_metrics(y_test, rf_model.predict(X_test), "Random Forest")

    xgb_model = train_xgboost(X_train, y_train)
    xgb_metrics = report_metrics(y_test, xgb_model.predict(X_test), "XGBoost")

    print()
    print(pd.DataFrame([lin_metrics, rf_metrics, xgb_metrics]).set_index("model").round(3))


if __name__ == "__main__":
    main()
