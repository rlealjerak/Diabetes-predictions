import sqlite3
import pandas as pd

conn = sqlite3.connect("db/diabetes_trends.db")

# Check if World Bank has data for the missing countries
print("=== World Bank raw data for missing countries ===")
print(pd.read_sql(
    "SELECT country_code, COUNT(*) as rows FROM raw_worldbank WHERE country_code IN ('USA','MEX','GBR','NGA') GROUP BY country_code",
    conn
))

# Check WHO raw data for same countries
print("\n=== WHO raw data for missing countries ===")
print(pd.read_sql(
    "SELECT spatial_dim, COUNT(*) as rows FROM raw_who_gho WHERE spatial_dim IN ('USA','MEX','GBR','NGA') GROUP BY spatial_dim",
    conn
))

# Check merged panel coverage
print("\n=== Merged panel coverage ===")
df = pd.read_parquet("data/processed/merged_panel.parquet")
for c in ["USA", "MEX", "GBR", "NGA"]:
    sub = df[df["iso3_code"] == c]
    if len(sub) > 0:
        coverage = sub.drop(columns=["iso3_code", "year"]).notna().mean()
        print(f"\n{c}: {len(sub)} rows")
        print(coverage)
    else:
        print(f"\n{c}: NOT FOUND")

# Check feature matrix
print("\n=== Feature matrix check ===")
feat = pd.read_parquet("data/model_ready/global_features.parquet")
for c in ["USA", "MEX", "GBR", "NGA"]:
    sub = feat[feat["iso3_code"] == c]
    print(f"{c}: {len(sub)} rows in feature matrix")

conn.close()
