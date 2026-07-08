"""Time-based train/test split, mirroring notebook section 6.

Splits on order_date_parsed rather than a random shuffle, since the data
spans ~8 weeks and a time-based split better matches how the model would
actually be deployed (train on past orders, evaluate on the next period).
"""

import pandas as pd

EXCLUDED_COLS = [
    "ID", "Delivery_person_ID", "Time_Order_picked", "Order_Date", "Time_Orderd",
    "Restaurant_latitude", "Restaurant_longitude",
    "Delivery_location_latitude", "Delivery_location_longitude",
    "Road_traffic_density", "Festival",
]
TARGET_COL = "Time_taken(min)"
NON_FEATURE_COLS = EXCLUDED_COLS + [TARGET_COL, "order_date_parsed"]


def get_feature_cols(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


def time_based_split(df: pd.DataFrame, test_quantile: float = 0.8):
    """Splits at the given quantile of order_date_parsed. Returns X_train, X_test, y_train, y_test."""
    split_date = df["order_date_parsed"].quantile(test_quantile)
    train_mask = df["order_date_parsed"] < split_date
    feature_cols = get_feature_cols(df)

    train_df, test_df = df[train_mask], df[~train_mask]
    X_train, y_train = train_df[feature_cols], train_df[TARGET_COL]
    X_test, y_test = test_df[feature_cols], test_df[TARGET_COL]
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    from data_loader import load_raw
    from clean import clean
    from features import add_features

    df = add_features(clean(load_raw()))
    X_train, X_test, y_train, y_test = time_based_split(df)
    print(f"train: {X_train.shape}, test: {X_test.shape}")
    print(f"train target mean: {y_train.mean():.2f}, test target mean: {y_test.mean():.2f}")
