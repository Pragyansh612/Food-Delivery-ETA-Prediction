"""Trains Random Forest and XGBoost with light grid search tuning, mirroring
notebook section 8. Prints the same comparison table as the notebook.
"""

import sys
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).parent))
from clean import clean
from data_loader import load_raw
from evaluate import report_metrics
from features import add_features
from split import time_based_split

RANDOM_STATE = 42

RF_PARAM_GRID = {"n_estimators": [100, 200], "max_depth": [8, 12, None]}
XGB_PARAM_GRID = {"n_estimators": [200, 400], "max_depth": [4, 6], "learning_rate": [0.05, 0.1]}


def train_random_forest(X_train, y_train):
    search = GridSearchCV(
        RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
        RF_PARAM_GRID, cv=3, scoring="neg_mean_absolute_error", n_jobs=-1,
    )
    search.fit(X_train, y_train)
    print("best RF params:", search.best_params_)
    return search.best_estimator_


def train_xgboost(X_train, y_train):
    search = GridSearchCV(
        XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1, objective="reg:squarederror"),
        XGB_PARAM_GRID, cv=3, scoring="neg_mean_absolute_error", n_jobs=-1,
    )
    search.fit(X_train, y_train)
    print("best XGB params:", search.best_params_)
    return search.best_estimator_


def main():
    df = add_features(clean(load_raw()))
    X_train, X_test, y_train, y_test = time_based_split(df)

    rf_model = train_random_forest(X_train, y_train)
    report_metrics(y_test, rf_model.predict(X_test), "Random Forest")

    xgb_model = train_xgboost(X_train, y_train)
    report_metrics(y_test, xgb_model.predict(X_test), "XGBoost")

    return rf_model, xgb_model


if __name__ == "__main__":
    main()
