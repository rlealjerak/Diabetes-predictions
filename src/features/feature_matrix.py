import sqlite3
import pandas as pd
import os 

# Load cleaned data from the database 
# Create the output file 
db_path = "db/diabetes_trends.db"
output_path = "data/model_ready/global_features.parquet"

df = pd.read_parquet('data/processed/merged_panel.parquet')

# Make sure that the data is sorted 
df = df.sort_values(["iso3_code", "year"]) 

# Create lag features for each indicator
for col, lag, name in [ 
    ("mean_bmi", 5, "bmi_lag_5"),
    ("mean_bmi", 10, "bmi_lag_10"),
    ("raised_blood_glucose_pct", 5, "glucose_pct_lag_5"),
    ("physical_inactivity_pct", 5, "inactivity_pct_lag_5"),  
    ("physical_inactivity_pct", 10, "inactivity_pct_lag_10"),
    ("urban_pop_pct", 5, "urban_pop_lag_5yr"),
    ("urban_pop_pct", 10, "urban_pop_lag_10yr"),
    ("health_exp_per_capita", 5, "health_exp_per_capita_lag_5yr"),
]: 
    lagged = df[["iso3_code", "year", col]].copy()
    lagged["year"] = lagged["year"] + lag
    lagged = lagged.rename(columns={col: name})
    df = df.merge(lagged, on=["iso3_code", "year"], how="left")

# Create derived features
# Rate of change 
df["bmi_5yr_change"] = df["mean_bmi"] - df["bmi_lag_5"] 
df["physical_inactivity_5yr_change"] = df["physical_inactivity_pct"] - df["inactivity_pct_lag_5"]
df["health_exp_relative"] = df["health_exp_per_capita"] / df["gdp_per_capita_ppp"] # Health expenditure relative to GDP per capita 

# Urban population 10yr change 
urban_lag = df[["iso3_code", "year", "urban_pop_pct"]].copy()
urban_lag["year"] = urban_lag["year"] + 10 
urban_lag = urban_lag.rename(columns={"urban_pop_pct": "urban_pop_pct_lag_10yr" }) 
df = df.merge(urban_lag, on=["iso3_code", "year"], how="left")
df["urban_pop_change_10yr"] = df["urban_pop_pct"] - df["urban_pop_pct_lag_10yr"]

# Interactions between indicators 
df["glucose_bmi_interaction"] = df["raised_blood_glucose_pct"] * df["mean_bmi"]
df["inactivity_bmi_interaction"] = df["physical_inactivity_pct"] * df["mean_bmi"]   

# Drop unnecessary rows from the feature matrix 
# Drop diabeted prevalence rows with missing target variable (diabetes_prevalence) value since we can't train on those 
df = df.dropna(subset=["diabetes_prev_agestd"])

# Drop rows with too many missing features (e.g. more than 50% missing) 
feature_cols = [c for c in df.columns if c not in ("iso3_code", "year", "diabetes_prev_agestd")]                                                         
df = df[df[feature_cols].notna().mean(axis=1) >= 0.5] 

#Save the matrix into a parquet file for modeling
df.to_parquet(output_path, index=False)
                                                                                                                                                           
print(f"Feature matrix: {len(df)} rows, {df['iso3_code'].nunique()} countries, "                                                                         
    f"years {df['year'].min()}-{df['year'].max()}")                                                                                                    
print(f"Columns: {list(df.columns)}")                                                                                                                    
print(f"Null counts:\n{df[feature_cols].isnull().sum()}")