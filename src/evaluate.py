"""Shared metric reporting, used by every model script."""

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def report_metrics(y_true, y_pred, label: str) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2 = r2_score(y_true, y_pred)
    print(f"{label}: MAE={mae:.3f}  RMSE={rmse:.3f}  R2={r2:.3f}")
    return {"model": label, "MAE": mae, "RMSE": rmse, "R2": r2}
