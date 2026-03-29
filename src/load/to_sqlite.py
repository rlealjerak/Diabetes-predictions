import pyarrow.parquet as pq
import sqlite3
import pandas as pd 

# Load the clean parquet file and load it to SQLite database 
df = pd.read_parquet('data/processed/country_year_panel.parquet')
conn = sqlite3.connect('db/diabetes_trends.db')
df.to_sql('country_year_indicators', conn, if_exists='replace', index=False)

print(f"Loaded {len(df)} rows, {df['iso3_code'].nunique()} countries, years {df['year'].min()}-{df['year'].max()}")  