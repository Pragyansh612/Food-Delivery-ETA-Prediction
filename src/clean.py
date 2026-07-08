"""Cleans the raw food delivery dataframe.

Mirrors the cleaning section of notebooks/01_eda_and_modeling.ipynb. Every
drop/impute decision here is explained in the notebook -- this module just
makes the same logic reusable/importable.
"""

import numpy as np
import pandas as pd

INDIA_LAT_BOUNDS = (6, 38)
INDIA_LON_BOUNDS = (68, 98)


def _in_india_bounds(lat: pd.Series, lon: pd.Series) -> pd.Series:
    lat_ok = lat.between(*INDIA_LAT_BOUNDS)
    lon_ok = lon.between(*INDIA_LON_BOUNDS)
    not_placeholder = ~((lat.abs() < 1) & (lon.abs() < 1))
    return lat_ok & lon_ok & not_placeholder


def clean(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Parses raw strings, drops unrecoverable rows, imputes low-rate missingness.

    Drops (no safe imputation exists): invalid restaurant/delivery coordinates,
    out-of-range ratings (>5), missing order timestamps.
    Imputes (low missingness, looks random, not tied to target): age, ratings,
    multiple_deliveries (median); traffic, festival, city (mode).
    """
    df = df_raw.copy()

    str_cols = df.select_dtypes(include=["object", "str"]).columns
    for c in str_cols:
        df[c] = df[c].str.strip()
    df = df.replace("NaN", np.nan)

    df["Weatherconditions"] = df["Weatherconditions"].str.replace("conditions ", "", regex=False)
    df["Time_taken(min)"] = df["Time_taken(min)"].str.replace("(min) ", "", regex=False).astype(int)

    for c in ["Delivery_person_Age", "Delivery_person_Ratings", "multiple_deliveries"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    valid_coords = _in_india_bounds(
        df["Restaurant_latitude"], df["Restaurant_longitude"]
    ) & _in_india_bounds(df["Delivery_location_latitude"], df["Delivery_location_longitude"])
    df = df[valid_coords].copy()

    df = df[~(df["Delivery_person_Ratings"] > 5)].copy()

    df = df[~df["Time_Orderd"].isna()].copy()

    for c in ["Delivery_person_Age", "Delivery_person_Ratings", "multiple_deliveries"]:
        df[c] = df[c].fillna(df[c].median())
    for c in ["Road_traffic_density", "Festival", "City"]:
        df[c] = df[c].fillna(df[c].mode()[0])

    return df.reset_index(drop=True)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    from data_loader import load_raw

    raw = load_raw()
    cleaned = clean(raw)
    print(f"raw={len(raw)} -> cleaned={len(cleaned)} ({len(cleaned) / len(raw):.1%} kept)")
    print("missing values remaining:", cleaned.isna().sum().sum())
