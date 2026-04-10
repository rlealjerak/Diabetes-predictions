import pandas as pd 

# Load cleand data 
df_who = pd.read_parquet('data/processed/country_year_panel.parquet')
df_worldbank = pd.read_parquet('data/processed/worldbank_country_year_panel.parquet') 

merged = df_who.merge(df_worldbank, on=['iso3_code', 'year'], how='outer') 
merged = merged.sort_values(['iso3_code', 'year']) 

# Save merged data to a new parquet file 
merged.to_parquet('data/processed/merged_panel.parquet', index=False)                                                             
print(merged.shape)