CREATE TABLE IF NOT EXISTS dim_country (
    iso3_code    TEXT PRIMARY KEY,
    iso2_code    TEXT UNIQUE,
    country_name TEXT NOT NULL,
    region       TEXT,
    income_group TEXT
);

CREATE TABLE IF NOT EXISTS dim_indicator (
    indicator_id  TEXT PRIMARY KEY,
    source        TEXT NOT NULL,
    original_code TEXT,
    description   TEXT,
    unit          TEXT
);

CREATE TABLE IF NOT EXISTS raw_who_gho (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    indicator_code TEXT NOT NULL,
    spatial_dim    TEXT NOT NULL,
    time_dim       INTEGER NOT NULL,
    dim1           TEXT,
    numeric_value  REAL,
    low            REAL,
    high           REAL,
    ingested_at    TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS raw_worldbank (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    indicator_code TEXT NOT NULL,
    country_iso3   TEXT NOT NULL,
    year           INTEGER NOT NULL,
    value          REAL,
    ingested_at    TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS raw_owid (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    entity        TEXT NOT NULL,
    year          INTEGER NOT NULL,
    variable_name TEXT NOT NULL,
    value         REAL,
    ingested_at   TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS country_year_indicators (
    iso3_code                TEXT NOT NULL,
    year                     INTEGER NOT NULL,
    diabetes_prev_agestd     REAL,
    mean_bmi                 REAL,
    mean_fasting_glucose     REAL,
    raised_blood_glucose_pct REAL,
    physical_inactivity_pct  REAL,
    overweight_prev_pct      REAL,
    gdp_per_capita_ppp       REAL,
    urban_pop_pct            REAL,
    health_exp_per_capita    REAL,
    health_exp_pct_gdp       REAL,
    pop_age_65_plus_pct      REAL,
    life_expectancy          REAL,
    population               INTEGER,
    ncd_mortality_prob       REAL,
    PRIMARY KEY (iso3_code, year),
    FOREIGN KEY (iso3_code) REFERENCES dim_country(iso3_code)
);

CREATE TABLE IF NOT EXISTS simulation_results (
    simulation_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    scenario_name       TEXT NOT NULL,
    iso3_code           TEXT NOT NULL,
    target_year         INTEGER NOT NULL,
    baseline_prediction REAL,
    scenario_prediction REAL,
    delta               REAL,
    modified_features   TEXT,
    model_name          TEXT,
    created_at          TEXT DEFAULT (datetime('now'))
);
