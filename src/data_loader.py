"""Loads the raw food delivery CSV.

Source: Kaggle "Food Delivery Dataset" (gauravmalik26), pulled via a GitHub
mirror of the same file since Kaggle API credentials weren't available in
this environment. https://www.kaggle.com/datasets/gauravmalik26/food-delivery-dataset
"""

import pandas as pd

RAW_PATH = "data/raw/food_delivery_train.csv"


def load_raw(path: str = RAW_PATH) -> pd.DataFrame:
    """Reads the CSV as-is, no cleaning. Useful for EDA on the messy data."""
    return pd.read_csv(path)


if __name__ == "__main__":
    df = load_raw()
    print(f"loaded {df.shape[0]} rows, {df.shape[1]} cols")
    print(df.head(3))
