import pandas as pd
import numpy as np
import joblib
import sqlite3
from src.simulations.scenarios import SCENARIOS, TOP_COUNTRIES, PROJECTION_YEARS


def extrapolate_country(country_df, feature_cols, feature_bounds, projection_years):
    """Fit linear trend on last 10 years, project to future years."""
    recent = country_df.tail(10)
    rows = []
    for year in projection_years:
        row = {"iso3_code": country_df["iso3_code"].iloc[0], "year": year}
        for col in feature_cols:
            series = recent[["year", col]].dropna()
            if len(series) >= 2:
                coeffs = np.polyfit(series["year"], series[col], 1)
                row[col] = coeffs[0] * year + coeffs[1]
            else:
                row[col] = country_df[col].iloc[-1]
            row[col] = np.clip(row[col], feature_bounds.loc[col, 'min'], feature_bounds.loc[col, 'max'])
        rows.append(row)
    return pd.DataFrame(rows)


def run_simulations():
    model            = joblib.load("outputs/models/svm_model.pkl")
    scaler           = joblib.load("outputs/models/svm_scaler.pkl")
    training_medians = joblib.load("outputs/models/svm_training_medians.pkl")
    feature_bounds   = joblib.load("outputs/models/svm_feature_bounds.pkl")
    df               = pd.read_parquet("data/model_ready/global_features.parquet")

    selected_features = list(training_medians.index)

    results = []

    for country in TOP_COUNTRIES:
        country_df = df[df["iso3_code"] == country].sort_values("year")
        if country_df.empty:
            continue

        projected = extrapolate_country(country_df, selected_features, feature_bounds, PROJECTION_YEARS)

        for scenario_name, multipliers in SCENARIOS.items():
            scenario_df = projected.copy()

            for feature, mult in multipliers.items():
                if feature in scenario_df.columns:
                    scenario_df[feature] = scenario_df[feature] * mult

            if "mean_bmi" in multipliers and "bmi_5yr_change" in scenario_df.columns:
                scenario_df["bmi_5yr_change"] = scenario_df["bmi_5yr_change"] * multipliers["mean_bmi"]

            X = scenario_df[selected_features].fillna(training_medians)
            X_scaled = scaler.transform(X)

            preds = np.clip(model.predict(X_scaled), 0, 50)

            for i, year in enumerate(PROJECTION_YEARS):
                results.append({
                    "iso3_code":            country,
                    "year":                 year,
                    "scenario":             scenario_name,
                    "predicted_prevalence": round(preds[i], 2),
                })

    results_df = pd.DataFrame(results)

    conn = sqlite3.connect("db/diabetes_trends.db")
    results_df.to_sql("simulation_results", conn, if_exists="replace", index=False)
    conn.close()

    results_df.to_csv("outputs/reports/simulation_results.csv", index=False)

    pd.set_option('display.max_columns', None)
    print(results_df.pivot_table(index=["iso3_code", "year"], columns="scenario", values="predicted_prevalence"))

    return results_df


if __name__ == "__main__":
    run_simulations()
