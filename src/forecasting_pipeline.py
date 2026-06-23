"""Forecasting pipeline script

Usage:
    python src/forecasting_pipeline.py --data data/sample_sales.csv --output outputs

This script:
- loads a CSV with columns: date, sales
- performs basic cleaning and time-based feature engineering
- trains a RandomForestRegressor baseline
- evaluates (MAE, RMSE)
- saves a forecast plot to outputs/forecast.png and model to outputs/model.joblib
"""

import argparse
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

sns.set(style="whitegrid")


def load_data(path):
    df = pd.read_csv(path, parse_dates=["date"]) 
    df = df.sort_values("date").reset_index(drop=True)
    return df


def feature_engineer(df):
    df = df.copy()
    df["day"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["weekday"] = df["date"].dt.weekday
    df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
    df["is_month_end"] = df["date"].dt.is_month_end.astype(int)

    # lags
    for lag in [1, 7, 14]:
        df[f"lag_{lag}"] = df["sales"].shift(lag)

    # rolling
    df["roll_7"] = df["sales"].rolling(window=7, min_periods=1).mean().shift(1)
    df["roll_14"] = df["sales"].rolling(window=14, min_periods=1).mean().shift(1)

    df = df.dropna().reset_index(drop=True)
    return df


def train_and_evaluate(df, output_dir):
    features = [c for c in df.columns if c not in ("date", "sales")]
    X = df[features]
    y = df["sales"]

    # time-aware split: train on first 80%, test last 20%
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    print(f"MAE: {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")

    # Save model
    joblib.dump(model, os.path.join(output_dir, "model.joblib"))

    # Forecast (predict on the test set dates)
    df_pred = df.iloc[split_idx:].copy()
    df_pred["preds"] = preds

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(df["date"], df["sales"], label="actual", marker="o")
    plt.plot(df_pred["date"], df_pred["preds"], label="forecast", marker="o")
    plt.axvline(df_pred["date"].iloc[0], color="gray", linestyle="--", alpha=0.7)
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.title("Sales and Forecast")
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "forecast.png")
    plt.savefig(plot_path)
    print(f"Saved forecast plot to {plot_path}")

    # Save evaluation metrics
    with open(os.path.join(output_dir, "metrics.txt"), "w") as f:
        f.write(f"MAE: {mae:.6f}\n")
        f.write(f"RMSE: {rmse:.6f}\n")

    print(f"Model saved to {os.path.join(output_dir, 'model.joblib')}")

    return model, mae, rmse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to CSV with date and sales columns")
    parser.add_argument("--output", default="outputs", help="Output directory")
    args = parser.parse_args()

    output_dir = args.output
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    df = load_data(args.data)
    df_fe = feature_engineer(df)
    train_and_evaluate(df_fe, output_dir)


if __name__ == "__main__":
    main()
