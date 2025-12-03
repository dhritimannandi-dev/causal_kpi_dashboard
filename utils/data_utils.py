import pandas as pd
import json
import os


def load_results(results_path: str):
    with open(results_path, "r") as f:
        return json.load(f)


def load_analysis_report(report_path: str):
    with open(report_path, "r") as f:
        return f.read()


def load_timeseries(timeseries_path: str):
    df = pd.read_csv(timeseries_path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df

