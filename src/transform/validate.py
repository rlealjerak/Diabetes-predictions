import pandera as pa
import pandas as pd 

# Define a schema for the cleaned data
schema = pa.DataFrameSchema({
      "iso3_code": pa.Column(str, pa.Check(lambda s: s.str.len() <= 6)),
      "year": pa.Column(int, pa.Check.in_range(1990, 2025)),                                                                                               
      "diabetes_prev_agestd": pa.Column(float, pa.Check.in_range(0, 50), nullable=True),                                                                   
      "mean_bmi": pa.Column(float, pa.Check.in_range(15, 45), nullable=True),                                                                              
      "raised_blood_glucose_pct": pa.Column(float, pa.Check.in_range(0, 100), nullable=True),                                                              
      "physical_inactivity_pct": pa.Column(float, pa.Check.in_range(0, 100), nullable=True),                                                               
      "ncd_mortality_prob": pa.Column(float, pa.Check.in_range(0, 100), nullable=True),   
}) 

df = pd.read_parquet('data/processed/country_year_panel.parquet')

# Validate the Data against the schema
try: 
    schema.validate(df, lazy=True)
    print("Validation passed")
except pa.errors.SchemaErrors as e:
    print(e.failure_cases)

