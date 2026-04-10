import pandas as pd 
import numpy as np 
import joblib 
import sqlite3
from src.simulations.scenarios import SCENARIOS, TOP_COUNTRIES, PROJECTION_YEARS 

# Load the SVM model and its scaler 
model = joblib.load("outputs/models/svm_model.pkl")
scaler = joblib.load("outputs/models/svm_scaler.pkl") 

# Load the feature matrix 
df = pd.read_parquet("data/model_ready/global_features.parquet") 

# Get feature columns 
feature_cols = [c for c in df.columns if c not in ("iso3_code", "year", "diabetes_prev_agestd")]

# Extrapolate trends per country 
def extrapolate_country(country_df, feature_cols, projection_years):
    """Fit linear trend on last 10 years, project to future years"""
    recent = country_df.tail(10)
    rows = [] 
    for year in projection_years: 
        row = { "iso3_code": country_df["iso3_code"].iloc[0], "year": year}
        for col in feature_cols : 
            series = recent[["year", col]].dropna() 
            if len(series) >= 2: 
                # Fit linear trend 
                coeffs = np.polyfit(series["year"], series[col], 1 )
                row[col] = coeffs[0] * year + coeffs[1]
            else: 
                row[col] = country_df[col].iloc[-1] # fallback: last known value
        rows.append(row)
    return pd.DataFrame(rows)  
    
# Run simulations per country and scenario 
results = [] 

for country in TOP_COUNTRIES: 
    country_df = df[df["iso3_code"] == country].sort_values("year")
    if country_df.empty: 
        continue 

    # Extrapolate features to future years we want to predict on 
    projected = extrapolate_country(country_df, feature_cols, PROJECTION_YEARS)

    for scenario_name, multipliers in SCENARIOS.items():
        scenario_df = projected.copy()

        # Apply multipliers to each feature 
        for feature, mult in multipliers.items():
            if feature in scenario_df.columns:
                scenario_df[feature] = scenario_df[feature] * mult 
        
        # Recalculate derived features after applying the multipliers
        if "bmi_lag_5" in scenario_df.columns and "mean_bmi" in multipliers: 
            scenario_df["bmi_lag_5"] = scenario_df["bmi_lag_5"] * multipliers["mean_bmi"] 
            scenario_df["bmi_lag_10"] = scenario_df["bmi_lag_10"] * multipliers["mean_bmi"] 
            scenario_df["bmi_5y_change"] = scenario_df["mean_bmi"] - scenario_df["bmi_lag_5"] 
        if "glucose_bmi_interaction" in scenario_df.columns: 
            scenario_df["glucose_bmi_interaction"] = scenario_df["raised_blood_glucose_pct"] * scenario_df["mean_bmi"] 
            scenario_df["inactivity_bmi_interaction"] = scenario_df["physical_inactivity_pct"] * scenario_df["mean_bmi"] 

        # Handle null values and scale feature 
        X = scenario_df[feature_cols].fillna(0)
        X_scaled = scaler.transform(X) 

        # Predict diabetes prevalence
        preds = model.predict(X_scaled)
        preds = np.clip(preds, 0, 50)  # Ensure predictions are within a reasonable range 

        for i, year in enumerate( PROJECTION_YEARS): 
            results.append({
                "iso3_code": country, 
                "year": year, 
                "scenario": scenario_name, 
                "predicted_prevalence": round(preds[i], 2),
            })
    
# Save results of the simulations 
results_df = pd.DataFrame(results) 

# Save to SQLite 
conn = sqlite3.connect("db/diabetes_trends.db")
results_df.to_sql("simulation_results", conn, if_exists="replace", index=False)
conn.close()

# Save to CSV 
results_df.to_csv("outputs/reports/simulation_results.csv", index=False) 

pd.set_option('display.max_columns', None)  
print(results_df.pivot_table(index=["iso3_code", "year"], columns="scenario", values="predicted_prevalence"))