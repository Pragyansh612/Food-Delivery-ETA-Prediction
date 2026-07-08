"""Trains the Linear Regression baseline, mirroring notebook section 7."""

import sys
from pathlib import Path

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from clean import clean
from data_loader import load_raw
from evaluate import report_metrics
from features import add_features
from split import time_based_split


def main():
    df = add_features(clean(load_raw()))
    X_train, X_test, y_train, y_test = time_based_split(df)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)

    return report_metrics(y_test, preds, "Linear Regression")


if __name__ == "__main__":
    main()
