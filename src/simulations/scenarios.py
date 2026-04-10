SCENARIOS = {
    "baseline": {},  #no changes, just extrapolated trends 

    "reduce_bmi_5pct": {
        "mean_bmi": 0.95,
    }, 

    "reduce_glucose_10pct": { 
        "raised_blood_glucose_pct": 0.90,
    },  

    "increase_physical_activity": {
        "physical_inactivity_pct": 0.70,  #30% reduction in physical inactivity 
    },

    "combined_intervention": {
        "mean_bmi": 0.95,
        "raised_blood_glucose_pct": 0.90,
        "physical_inactivity_pct": 0.70, 
    }, 

    "worst_case": {
        "mean_bmi": 1.05,
        "physical_inactivity_pct": 1.20,  #20% increase in physical inactivity
    },

    "increase_health_expenditure_per_capita": {
        "health_exp_per_capita": 1.25,  #25% increase in health expenditure per capita
    },
}

TOP_COUNTRIES = ["USA", "IND", "CHN", "BRA", "MEX", "IDN", "DEU", "GBR", "JPN", "NGA"] 
PROJECTION_YEARS = [2030, 2040, 2050]

