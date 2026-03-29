import sqlite3
import pandas as pd 

conn = sqlite3.connect('db/diabetes_trends.db')
df = pd.read_sql('SELECT * FROM raw_who_gho', conn)


# Pivot indicators into columns 
df_wide = df.pivot_table( 
    index=['spatial_dim', 'time_dim', 'dim1'],
    columns='indicator_code',
    values='numeric_value'
).reset_index() 

# Rename columns to match the indicators 
df_wide.rename(columns={
      'spatial_dim': 'iso3_code',
      'time_dim': 'year',
      'NCD_DIABETES_PREVALENCE_AGESTD': 'diabetes_prev_agestd',
      'NCD_BMI_MEAN': 'mean_bmi',
      'NCD_GLUC_04': 'raised_blood_glucose_pct',
      'NCD_PAA': 'physical_inactivity_pct',
      'NCDMORT3070': 'ncd_mortality_prob'
}, inplace=True)

# Drop dim1 column 
df_wide = df_wide.drop(columns=['dim1'])

# Perform linear interpolation to fill in missing values for each indicator
df_wide = df_wide.sort_values(['iso3_code', 'year'])
df_wide = df_wide.groupby('iso3_code').apply(lambda group: group.interpolate(method='linear', limit=2)).reset_index(drop=True)

# Exclude rows were the year < 1990 
df_wide = df_wide[df_wide['year'] >= 1990]

# Save cleaned data to parquet
df_wide.to_parquet('data/processed/country_year_panel.parquet', index=False)                                                                             
print(df_wide.shape)