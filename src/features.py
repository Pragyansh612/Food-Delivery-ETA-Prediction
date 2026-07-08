"""Feature engineering, mirroring notebook section 4.

Takes the cleaned dataframe from clean.clean() and returns a model-ready
dataframe: distance, time-of-day features, encoded categoricals, and one
justified interaction term.
"""

import numpy as np
import pandas as pd

TRAFFIC_ORDER = {"Low": 0, "Medium": 1, "High": 2, "Jam": 3}
ONEHOT_COLS = ["Weatherconditions", "Type_of_order", "Type_of_vehicle", "City"]
ONEHOT_PREFIXES = ["weather", "order_type", "vehicle", "city"]


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["distance_km"] = haversine_km(
        df["Restaurant_latitude"], df["Restaurant_longitude"],
        df["Delivery_location_latitude"], df["Delivery_location_longitude"],
    )

    order_datetime = pd.to_datetime(df["Order_Date"], format="%d-%m-%Y")
    order_time = pd.to_datetime(df["Time_Orderd"], format="%H:%M:%S")
    df["order_hour"] = order_time.dt.hour
    df["order_day_of_week"] = order_datetime.dt.dayofweek
    df["is_weekend"] = df["order_day_of_week"].isin([5, 6]).astype(int)
    df["order_date_parsed"] = order_datetime

    df["traffic_ordinal"] = df["Road_traffic_density"].map(TRAFFIC_ORDER)
    df["festival_flag"] = (df["Festival"] == "Yes").astype(int)

    df = pd.get_dummies(df, columns=ONEHOT_COLS, prefix=ONEHOT_PREFIXES)

    df["distance_x_traffic"] = df["distance_km"] * df["traffic_ordinal"]

    return df


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    from data_loader import load_raw
    from clean import clean

    df = add_features(clean(load_raw()))
    print(df.shape)
    print(df.filter(like="weather_").columns.tolist())
