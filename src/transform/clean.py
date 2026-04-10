import sqlite3
import pandas as pd 
import pycountry 

conn = sqlite3.connect('db/diabetes_trends.db')
df = pd.read_sql('SELECT * FROM raw_who_gho', conn)
df_worldbank = pd.read_sql('SELECT * FROM raw_worldbank', conn)

# Drop agregates from worldbank data 
valid_iso3_codes = {c.alpha_3 for c in pycountry.countries}
df_worldbank = df_worldbank[df_worldbank['country_code'].isin(valid_iso3_codes)]

# Pivot indicators into columns (WORLDBANK dataset)
df_worldbank_wide = df_worldbank.pivot_table(
    index=['country_code', 'year'],
    columns='indicator_code',
    values='value'
).reset_index()
# Rename columns to something readable (WORLDBANK dataset)
df_worldbank_wide.rename(columns={                                                                                                       
      'country_code': 'iso3_code',
      'SH.STA.OWAD.ZS': 'overweight_pct',                                                                                           
      'NY.GDP.PCAP.PP.CD': 'gdp_per_capita_ppp',
      'SP.URB.TOTL.IN.ZS': 'urban_pop_pct',                                                                                         
      'SH.XPD.CHEX.PC.CD': 'health_exp_per_capita',
      'SH.XPD.CHEX.GD.ZS': 'health_exp_pct_gdp',                                                                                    
      'SP.POP.65UP.TO.ZS': 'pop_65plus_pct',                                                                                        
      'SP.DYN.LE00.IN': 'life_expectancy',                                                                                          
      'SP.POP.TOTL': 'population'                                                                                                   
  }, inplace=True)                 

# Pivot indicators into columns (WHO dataset)
df_wide = df.pivot_table( 
    index=['spatial_dim', 'time_dim', 'dim1'],
    columns='indicator_code',
    values='numeric_value'
).reset_index() 

# Rename columns to match the indicators (WHO dataset) 
df_wide.rename(columns={
      'spatial_dim': 'iso3_code',
      'time_dim': 'year',
      'NCD_DIABETES_PREVALENCE_AGESTD': 'diabetes_prev_agestd',
      'NCD_BMI_MEAN': 'mean_bmi',
      'NCD_GLUC_04': 'raised_blood_glucose_pct',
      'NCD_PAA': 'physical_inactivity_pct',
      'NCDMORT3070': 'ncd_mortality_prob'
}, inplace=True)

# Drop dim1 column (WHO dataset)
df_wide = df_wide.drop(columns=['dim1'])

# Perform linear interpolation to fill in missing values for each indicator (WHO dataset) 
df_wide = df_wide.sort_values(['iso3_code', 'year'])
df_wide = df_wide.groupby('iso3_code').apply(lambda group: group.interpolate(method='linear', limit=2)).reset_index(drop=True)

# Exclude rows were the year < 1990 (WHO dataset) 
df_wide = df_wide[df_wide['year'] >= 1990]

# Save cleaned data to parquet (WHO dataset) 
df_wide.to_parquet('data/processed/country_year_panel.parquet', index=False)                                                                             
print(df_wide.shape)

# Save cleaned data to parquet (WORLDBANK dataset)
df_worldbank_wide.to_parquet('data/processed/worldbank_country_year_panel.parquet', index=False)
print(df_worldbank_wide.shape)