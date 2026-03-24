# Global Diabetes Futures — Project Blueprint

## Context

**Problem:** Diabetes prevalence is projected to rise sharply through 2050, but the relative influence of modifiable risk factors (obesity, inactivity, diet) on that trajectory is not well quantified in an accessible, reproducible way.

**Goal:** Build a portfolio-quality end-to-end system that ingests multi-source public health data, runs it through a validated ETL pipeline into a database, trains interpretable models to predict country-level diabetes prevalence, and runs what-if simulations to estimate how much reducing key risk factors could change outcomes by 2050.

**Why it works for a portfolio:** Demonstrates SQL schema design, API ingestion, data validation, pipeline orchestration, predictive modeling, SHAP-based interpretability, and scenario simulation — all in one coherent project.

---

## 1. Architecture Overview

```
                    DATA SOURCES
    ┌─────────┬──────────┬───────────┬──────────┐
    │ WHO GHO │World Bank│  OWID/    │CDC BRFSS │
    │  API    │   API    │NCD-RisC   │  (CSV)   │
    └────┬────┴────┬─────┴─────┬─────┴────┬─────┘
         │         │           │          │
         v         v           v          v
    ┌─────────────────────────────────────────┐
    │         EXTRACTION LAYER                │
    │  (Python scripts per source, idempotent) │
    └──────────────────┬──────────────────────┘
                       v
    ┌─────────────────────────────────────────┐
    │         RAW LAYER (SQLite)              │
    │  raw_who, raw_worldbank, raw_owid, etc. │
    │  (append-only, timestamped)             │
    └──────────────────┬──────────────────────┘
                       v
    ┌─────────────────────────────────────────┐
    │       TRANSFORM + VALIDATE LAYER        │
    │  pandera schemas, ISO normalization,    │
    │  cleaning, interpolation                │
    └──────────────────┬──────────────────────┘
                       v
    ┌─────────────────────────────────────────┐
    │         PROCESSED LAYER (SQLite)        │
    │  country_year_indicators (unified)      │
    │  us_state_year_indicators               │
    └──────────────────┬──────────────────────┘
                       v
    ┌──────────────────┴──────────────────────┐
    v                                          v
┌───────────────┐                  ┌───────────────────┐
│  DASHBOARDS   │                  │  MODELS + SIMS    │
│  (Streamlit)  │                  │  (sklearn, SHAP)  │
└───────────────┘                  └───────────────────┘
```

---

## 2. Datasets & Variables

### 2.1 WHO Global Health Observatory (GHO) — Primary diabetes source
- **Access:** REST API `https://ghoapi.azureedge.net/api/{IndicatorCode}` — no auth required
- **Granularity:** Country-year (~1990–2022), filter to `Dim1='BTSX'` (both sexes)
- **Variables:**

| Variable | Indicator Code |
|---|---|
| Diabetes prevalence (age-std) | `NCD_DIABETES_PREVALENCE_AGESTD` |
| Mean BMI (age-std) | `NCD_BMI_MEAN` |
| Mean fasting blood glucose | `NCD_GLUC_01` |
| Raised blood glucose % | `NCD_GLUC_04` |
| Physical inactivity % | `NCD_PAA` |
| NCD mortality probability | `NCDMORT3070` |

### 2.2 World Bank World Development Indicators — Socioeconomic context
- **Access:** REST API `https://api.worldbank.org/v2/country/all/indicator/{code}?format=json` — no auth
- **Variables:**

| Variable | Indicator Code |
|---|---|
| Overweight prevalence, adults | `SH.STA.OWAD.ZS` |
| GDP per capita (PPP) | `NY.GDP.PCAP.PP.CD` |
| Urban population % | `SP.URB.TOTL.IN.ZS` |
| Health expenditure per capita | `SH.XPD.CHEX.PC.CD` |
| Health expenditure % GDP | `SH.XPD.CHEX.GD.ZS` |
| Population 65+ % | `SP.POP.65UP.TO.ZS` |
| Life expectancy | `SP.DYN.LE00.IN` |
| Population total | `SP.POP.TOTL` |

### 2.3 Our World in Data (OWID) / NCD-RisC — Gap filler
- **Access:** CSV download from GitHub (`ourworldindata.org/diabetes`, `ourworldindata.org/obesity`)
- **Variables:** BMI trends by sex, diabetes share, supplemental indicators

### 2.4 IDF Diabetes Atlas — Validation benchmark (extended)
- **Access:** Manual download from `diabetesatlas.org/data/en/`
- **Key use:** Compare your 2030/2045 projections against IDF's published projections

### 2.5 CDC BRFSS / Chronic Disease Indicators — US depth (extended)
- **Access:** Socrata API `https://data.cdc.gov/resource/fn2i-3j6c.json`
- **Variables:** State-level diabetes, obesity, inactivity, smoking prevalence

### Join Strategy
- **Global:** Join on `(iso3_code, year)` — WHO uses ISO3 natively, World Bank uses ISO3 in `country.id`, OWID uses country names → map via `pycountry`
- **US:** Join on `(state_name, year)` or `(state_fips, year)`

---

## 3. Folder Structure

```
diabetes-futures/
├── README.md
├── Makefile                    # Orchestration: make extract, make transform, make all
├── pyproject.toml
├── .env.example
├── .gitignore
├── data/
│   ├── raw/                    # Immutable API/CSV extracts
│   │   ├── who/
│   │   ├── worldbank/
│   │   ├── owid/
│   │   ├── idf/
│   │   └── cdc_brfss/
│   ├── processed/              # Cleaned, merged intermediates
│   │   └── country_year_panel.parquet
│   └── model_ready/            # Feature matrices
│       └── global_features.parquet
├── db/
│   ├── diabetes_trends.db      # SQLite database
│   └── schema.sql              # DDL
├── src/
│   ├── __init__.py
│   ├── extract/                # One module per source
│   │   ├── who_gho.py
│   │   ├── worldbank.py
│   │   ├── owid_csv.py
│   │   ├── idf_atlas.py
│   │   └── cdc_brfss.py
│   ├── transform/
│   │   ├── clean.py            # Nulls, outliers, dtype coercion
│   │   ├── normalize.py        # ISO code alignment, unit standardization
│   │   ├── merge.py            # Join sources into unified panel
│   │   └── validate.py         # pandera schemas
│   ├── load/
│   │   └── to_sqlite.py        # DataFrame → SQLite
│   ├── features/
│   │   └── engineer.py         # Lags, rates of change, interactions
│   ├── models/
│   │   ├── baseline.py         # Linear, Ridge
│   │   ├── advanced.py         # Random Forest, XGBoost
│   │   └── evaluate.py         # Metrics, cross-validation
│   ├── simulation/
│   │   ├── scenarios.py        # Scenario definitions
│   │   └── engine.py           # Modify features → re-predict
│   └── explainability/
│       └── shap_analysis.py    # SHAP, PDP, permutation importance
├── notebooks/
│   ├── 01_eda_global.ipynb
│   ├── 02_modeling.ipynb
│   ├── 03_simulations.ipynb
│   └── 04_explainability.ipynb
├── tests/
│   ├── test_extract.py
│   ├── test_transform.py
│   └── test_validate.py
├── app/
│   └── streamlit_app.py        # Interactive dashboard (extended)
└── outputs/
    ├── figures/
    └── reports/
```

---

## 4. Database Schema (SQLite)

### Reference Tables
```sql
CREATE TABLE dim_country (
    iso3_code    TEXT PRIMARY KEY,
    iso2_code    TEXT UNIQUE,
    country_name TEXT NOT NULL,
    region       TEXT,
    income_group TEXT
);

CREATE TABLE dim_indicator (
    indicator_id  TEXT PRIMARY KEY,
    source        TEXT NOT NULL,
    original_code TEXT,
    description   TEXT,
    unit          TEXT
);
```

### Raw Layer (one table per source, append-only)
```sql
CREATE TABLE raw_who_gho (
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

CREATE TABLE raw_worldbank (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    indicator_code TEXT NOT NULL,
    country_iso3   TEXT NOT NULL,
    year           INTEGER NOT NULL,
    value          REAL,
    ingested_at    TEXT DEFAULT (datetime('now'))
);

CREATE TABLE raw_owid (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    entity        TEXT NOT NULL,
    year          INTEGER NOT NULL,
    variable_name TEXT NOT NULL,
    value         REAL,
    ingested_at   TEXT DEFAULT (datetime('now'))
);
```

### Processed Layer (unified, clean)
```sql
CREATE TABLE country_year_indicators (
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
```

### Simulation Results
```sql
CREATE TABLE simulation_results (
    simulation_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    scenario_name       TEXT NOT NULL,
    iso3_code           TEXT NOT NULL,
    target_year         INTEGER NOT NULL,
    baseline_prediction REAL,
    scenario_prediction REAL,
    delta               REAL,
    modified_features   TEXT,   -- JSON
    model_name          TEXT,
    created_at          TEXT DEFAULT (datetime('now'))
);
```

---

## 5. ETL Pipeline Design

### Extraction (`src/extract/`)
Each extractor has a `run()` function that:
1. Calls the API or reads CSV
2. Saves raw data to `data/raw/{source}/`
3. Inserts into `raw_*` SQLite table
4. Is **idempotent** (`INSERT OR IGNORE` / check `ingested_at`)

**WHO pattern:** `GET https://ghoapi.azureedge.net/api/{code}?$filter=Dim1 eq 'BTSX'`
**World Bank pattern:** `GET https://api.worldbank.org/v2/country/all/indicator/{code}?format=json&date=1990:2023&per_page=20000`

### Transformation (`src/transform/`)
1. **clean.py** — Drop World Bank aggregates (WLD, EAS, etc.), fix types, flag outliers (log, don't remove)
2. **normalize.py** — Map all sources to ISO3 using `pycountry`, standardize to both-sexes/18+ age group
3. **merge.py** — Build full `(iso3_code, year)` grid, left-join each source, interpolate gaps of 1–2 years linearly within each country
4. **validate.py** — pandera schemas enforce ranges (diabetes 0–50%, BMI 15–45, year 1990–2025)

### Load (`src/load/`)
- `to_sqlite.py` — `df.to_sql()` with `if_exists='replace'` for processed tables (idempotent full reload)

### Orchestration (Makefile)
```makefile
all: init extract transform load features train simulate

init:
	python -c "import sqlite3; sqlite3.connect('db/diabetes_trends.db').executescript(open('db/schema.sql').read())"
extract:
	python -m src.extract.who_gho
	python -m src.extract.worldbank
	python -m src.extract.owid_csv
transform:
	python -m src.transform.clean
	python -m src.transform.normalize
	python -m src.transform.merge
	python -m src.transform.validate
load:
	python -m src.load.to_sqlite
features:
	python -m src.features.engineer
train:
	python -m src.models.baseline
	python -m src.models.advanced
simulate:
	python -m src.simulation.engine
```

---

## 6. Feature Engineering

### Time-Based
- **Lag features:** `bmi_lag_5`, `bmi_lag_10` — diabetes develops over years of metabolic stress. `df.groupby('iso3_code')['mean_bmi'].shift(5)`
- **Rate of change:** `bmi_5yr_change = (bmi - bmi_lag_5) / 5`
- **Urbanization velocity:** `urban_pop_change_10yr`

### Interactions
- `glucose_bmi_interaction = mean_fasting_glucose * mean_bmi`
- `inactivity_obesity_interaction = physical_inactivity_pct * overweight_prev_pct`

### Categorical Encodings
- `income_group`: ordinal (Low=1, Lower-middle=2, Upper-middle=3, High=4)
- `region`: one-hot encode

### Derived Ratios
- `health_exp_relative = health_exp_per_capita / gdp_per_capita_ppp`

---

## 7. Model Progression

| Stage | Model | Purpose |
|---|---|---|
| Baseline | Pooled OLS Linear Regression | Establish baseline, understand linear relationships |
| Baseline+ | Ridge Regression | Handle collinearity with regularization |
| Intermediate | Random Forest | Capture non-linearities, quick feature importance |
| **Primary** | **XGBoost** | Best performance, primary model for simulations |
| Extended | Panel FE regression (statsmodels) | Econometrically proper for panel data |

### Evaluation
- **GroupKFold** by country (5 folds) — all years of a country stay in same fold
- **Temporal holdout:** Most recent 3 years (2020–2022) as test set
- **Metrics:** RMSE, MAE, R², MAPE

---

## 8. What-If Simulation

### Core Logic
A trained model is `f(features) → diabetes_prevalence`. To simulate:
1. Extrapolate features to 2050 per country (linear trend fit per feature)
2. Predict baseline: `f(projected_features)` → baseline diabetes prevalence
3. Apply scenario modifications (e.g., `overweight_prev_pct *= 0.80`)
4. Re-predict: `f(modified_features)` → scenario diabetes prevalence
5. Delta = scenario − baseline

### Defined Scenarios
| Scenario | Modifications |
|---|---|
| Reduce obesity 20% | overweight × 0.80, BMI × 0.97 |
| Increase physical activity | inactivity × 0.70 |
| Double health expenditure | health_exp × 2.0 |
| Combined intervention | All above combined |
| Worst-case obesity surge | overweight × 1.30, BMI × 1.04 |

### Validation
- Compare baseline 2030/2045 projections against IDF Diabetes Atlas published estimates
- Clamp outputs to 0–50% range

---

## 9. Explainability (3 complementary methods)

1. **Permutation importance** — `sklearn.inspection.permutation_importance` on test set
2. **SHAP values** — `shap.TreeExplainer(model)` → summary beeswarm, dependence plots, force plots for specific countries
3. **Partial Dependence Plots** — `sklearn.inspection.PartialDependenceDisplay` with ICE curves

### Deliverables
- Top-10 feature importance bar chart
- SHAP beeswarm summary plot (portfolio showpiece)
- SHAP dependence plots for top 3 features
- Per-country force plots (USA, India, Brazil)
- Written interpretation of findings

---

## 10. Python Libraries

```
pandas numpy requests pyarrow pandera pycountry
scikit-learn xgboost shap
matplotlib seaborn plotly streamlit
statsmodels pytest
```

---

## 11. Development Timeline

### MVP — Weeks 1–4

| Week | Focus | Deliverables |
|---|---|---|
| **1** | Setup + Extraction | Repo structure, SQLite schema, WHO + World Bank + OWID extractors working |
| **2** | Transform + Load | clean/normalize/merge/validate pipeline, `country_year_indicators` populated, EDA notebook |
| **3** | Features + Models | Feature matrix, OLS → Ridge → RF → XGBoost trained, metrics compared |
| **4** | Simulations + MVP ship | 5 scenarios × 10 countries, SHAP summary, README, push to GitHub |

### Extended — Weeks 5–8

| Week | Focus |
|---|---|
| **5** | CDC BRFSS ingestion, US state-level model + simulations |
| **6** | Full explainability suite, IDF validation comparison, bootstrap CIs |
| **7** | Streamlit dashboard (choropleth map, scenario sliders, SHAP waterfall) |
| **8** | Tests, docs, methodology write-up, demo video, portfolio polish |

---

## 12. MVP vs Extended Summary

| Component | MVP | Extended |
|---|---|---|
| Sources | WHO + World Bank + OWID (3) | + IDF + CDC BRFSS (5) |
| Granularity | Country-year | + US state-year |
| Models | OLS, Ridge, RF, XGBoost | + Panel FE regression |
| Simulations | 5 scenarios × 10 countries | All countries + US states |
| Explainability | Feature importance + SHAP summary | + PDP, force plots, per-country |
| Dashboard | Notebooks only | Streamlit app |
| Tests | None | pytest suite |

---

## 13. Verification Plan

1. **Pipeline:** Run `make all` end-to-end — should produce populated SQLite DB with no errors
2. **Data quality:** Check `country_year_indicators` has 150+ countries, 20+ years; pandera validation passes
3. **Models:** XGBoost test-set RMSE should be lower than OLS baseline; R² > 0.7 is a reasonable target
4. **Simulations:** Baseline 2030 projections should be within ±30% of IDF published estimates for major countries
5. **Explainability:** SHAP values should sum to (prediction − base_value) for each row (built-in SHAP property)
6. **Reproducibility:** Clone repo, run `pip install -r requirements.txt && make all` — should work from scratch

---

## 14. Risk Mitigation

| Risk | Mitigation |
|---|---|
| API downtime | Cache raw responses to `data/raw/`; pipeline reads from cache |
| Sparse data for some countries | Filter to countries with ≥5 data points; report coverage |
| Feature collinearity | Ridge regularization; SHAP handles correlated features |
| Projection uncertainty | Confidence intervals; frame as "scenario analysis" not "prediction" |
| Scope creep | MVP scoped to 4 weeks; extended features are additive |
