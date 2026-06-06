import os
import logging
import numpy as np
import pandas as pd

# Setup logging - Ember loves visible pipeline logs
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_raw_data(filepath):
    """Simulate data ingestion from SCADA / Excel logs."""
    if not os.path.exists(filepath):
        logging.warning(f"File {filepath} not found. Generating production-grade synthetic energy data...")
        dates = pd.date_range(start="2026-01-01", end="2026-05-31", freq="H")
        np.random.seed(42)
        df = pd.DataFrame({
            "timestamp": dates,
            "steam_consumption_m3": np.random.normal(loc=150, scale=25, size=len(dates)),
            "compressed_air_m3": np.random.normal(loc=500, scale=80, size=len(dates)),
            "electricity_kwh": np.random.normal(loc=1200, scale=150, size=len(dates))
        })
        # Inject anomalies to test the cleaning pipeline
        df.loc[10:12, "electricity_kwh"] = np.nan
        df.loc[100, "steam_consumption_m3"] = -999  # Outlier/Sensor error
        return df
    return pd.read_csv(filepath, parse_dates=["timestamp"])

def validate_and_clean_data(df):
    """Enforce strict industrial data quality rules before analytics."""
    cleaned_df = df.copy()
    
    # 1. Critical timestamp validation
    cleaned_df = cleaned_df.dropna(subset=["timestamp"])
    
    # 2. Sensor error mitigation (e.g., negative consumption is physically impossible)
    energy_cols = ["steam_consumption_m3", "compressed_air_m3", "electricity_kwh"]
    for col in energy_cols:
        initial_anomalies = (cleaned_df[col] < 0).sum()
        if initial_anomalies > 0:
            logging.info(f"Detected {initial_anomalies} negative sensor errors in {col}. Setting to NaN.")
            cleaned_df.loc[cleaned_df[col] < 0, col] = np.nan
            
    # 3. Data Imputation using forward fill (Standard for continuous timeseries)
    cleaned_df[energy_cols] = cleaned_df[energy_cols].ffill().bfill()
    
    logging.info("Data validation and cleaning pipeline completed successfully.")
    return cleaned_df

def calculate_energy_kpis(df):
    """Transform hourly operational timeseries into daily business metrics."""
    df["date"] = df["timestamp"].dt.date
    daily_summary = df.groupby("date").agg({
        "steam_consumption_m3": "sum",
        "compressed_air_m3": "sum",
        "electricity_kwh": "sum"
    }).reset_index()
    
    # 7-day moving average for smoothing seasonal factory shifts
    daily_summary["electricity_7d_moving_avg"] = daily_summary["electricity_kwh"].rolling(window=7, min_periods=1).mean()
    return daily_summary

if __name__ == "__main__":
    raw_data = load_raw_data("data/raw_factory_logs.csv")
    cleaned_data = validate_and_clean_data(raw_data)
    daily_kpis = calculate_energy_kpis(cleaned_data)
    
    os.makedirs("data", exist_ok=True)
    daily_kpis.to_csv("data/daily_energy_kpis.csv", index=False)
    logging.info("ETL Output successfully exported to 'data/daily_energy_kpis.csv'")
