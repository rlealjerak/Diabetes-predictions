# Global Diabetes Futures — Execution Plan

## Context
We have a fully designed project blueprint (see BLUEPRINT.md) for an end-to-end public health data pipeline that ingests WHO, World Bank, and OWID data, builds predictive models for country-level diabetes prevalence, and runs what-if simulations to 2050. This document is the step-by-step implementation guide.

---

## Phase 1: Project Scaffolding
**Goal:** Repo structure, dependencies, git init, database schema — everything needed before writing logic.

### Step 1.1 — Create folder structure & boilerplate
- Create all directories: `src/extract/`, `src/transform/`, `src/load/`, `src/features/`, `src/models/`, `src/simulation/`, `src/explainability/`, `data/raw/who/`, `data/raw/worldbank/`, `data/raw/owid/`, `data/raw/idf/`, `data/raw/cdc_brfss/`, `data/processed/`, `data/model_ready/`, `db/`, `notebooks/`, `tests/`, `app/`, `outputs/figures/`, `outputs/reports/`
- Create `__init__.py` files in `src/` and all sub-packages
- Create `.gitignore` (Python defaults + `data/raw/`, `db/*.db`, `.env`, `__pycache__/`, `*.pyc`, `.DS_Store`, `outputs/`)
- Create `.env.example` (empty, placeholder for future API keys)

### Step 1.2 — pyproject.toml
- Project metadata, Python >=3.10
- Dependencies: `pandas`, `numpy`, `requests`, `pyarrow`, `pandera`, `pycountry`, `scikit-learn`, `xgboost`, `shap`, `matplotlib`, `seaborn`, `plotly`, `streamlit`, `statsmodels`, `pytest`

### Step 1.3 — SQLite schema (`db/schema.sql`)
- `dim_country` — ISO3 primary key, region, income_group
- `dim_indicator` — indicator_id, source, original_code, description, unit
- `raw_who_gho` — append-only with ingested_at
- `raw_worldbank` — append-only with ingested_at
- `raw_owid` — append-only with ingested_at
- `country_year_indicators` — unified processed table, PK (iso3_code, year)
- `simulation_results` — scenario outputs with JSON modified_features
- All DDL exactly as specified in BLUEPRINT.md §4

### Step 1.4 — Makefile
- Targets: `init`, `extract`, `transform`, `load`, `features`, `train`, `simulate`, `all`
- `init` creates DB from schema.sql
- Each target runs the corresponding `python -m src.*` modules

### Step 1.5 — Git init + initial commit

**Verification:** `make init` creates `db/diabetes_trends.db` with all tables.

---

## Phase 2: Extraction Layer
**Goal:** Three working extractors that pull data from APIs/CSV into `data/raw/` and `raw_*` SQLite tables.

### Step 2.1 — `src/extract/who_gho.py`
- Indicators: `NCD_DIABETES_PREVALENCE_AGESTD`, `NCD_BMI_MEAN`, `NCD_GLUC_01`, `NCD_GLUC_04`, `NCD_PAA`, `NCDMORT3070`
- For each indicator: `GET https://ghoapi.azureedge.net/api/{code}?$filter=Dim1 eq 'BTSX'`
- Parse JSON response `.value` array
- Save raw JSON to `data/raw/who/{code}.json`
- Insert rows into `raw_who_gho` (INSERT OR IGNORE for idempotency)
- `run()` function + `if __name__ == '__main__': run()`

### Step 2.2 — `src/extract/worldbank.py`
- Indicators: `SH.STA.OWAD.ZS`, `NY.GDP.PCAP.PP.CD`, `SP.URB.TOTL.IN.ZS`, `SH.XPD.CHEX.PC.CD`, `SH.XPD.CHEX.GD.ZS`, `SP.POP.65UP.TO.ZS`, `SP.DYN.LE00.IN`, `SP.POP.TOTL`
- For each: `GET https://api.worldbank.org/v2/country/all/indicator/{code}?format=json&date=1990:2023&per_page=20000`
- World Bank returns `[metadata, data]` — use index [1]
- Handle pagination if needed (check metadata total/pages)
- Save raw JSON to `data/raw/worldbank/{code}.json`
- Insert into `raw_worldbank`

### Step 2.3 — `src/extract/owid_csv.py`
- Download diabetes CSV from OWID GitHub raw URL
- Download obesity/BMI CSV from OWID GitHub raw URL
- Save to `data/raw/owid/`
- Parse with pandas, insert into `raw_owid`

**Verification:** Run `make extract`. Check that `data/raw/` has JSON/CSV files. Query `SELECT COUNT(*) FROM raw_who_gho` etc. — should have thousands of rows.

---

## Phase 3: Transform + Validate Layer
**Goal:** Clean, normalize, merge, and validate data into a unified `country_year_indicators` table.

### Step 3.1 — `src/transform/clean.py`
- Read from `raw_*` SQLite tables
- WHO: extract `SpatialDim` (ISO3), `TimeDim` (year), `NumericValue`
- World Bank: drop aggregate codes (WLD, EAS, ECS, LCN, MEA, SAS, SSF, etc.), coerce types
- OWID: drop non-country entities (continents, "World")
- Flag outliers using IQR per indicator (log to console, don't remove)
- Output: cleaned DataFrames saved to `data/processed/` as intermediate parquet files

### Step 3.2 — `src/transform/normalize.py`
- Map all country identifiers to ISO3 using `pycountry`
  - WHO: already ISO3 in `SpatialDim`
  - World Bank: already ISO3 in `country.id`
  - OWID: country name → ISO3 via `pycountry.countries.search_fuzzy()`
- Standardize column names to match `country_year_indicators` schema
- Filter to both-sexes/18+ age groups where applicable

### Step 3.3 — `src/transform/merge.py`
- Build full `(iso3_code, year)` grid from the union of all countries × years (1990–2023)
- Left-join WHO indicators, World Bank indicators, OWID indicators
- Populate `dim_country` reference table (ISO3, name, region, income_group from World Bank metadata)
- Populate `dim_indicator` reference table
- Linear interpolation for gaps of 1–2 years within each country: `df.groupby('iso3_code').apply(lambda g: g.interpolate(method='linear', limit=2))`
- Output: `data/processed/country_year_panel.parquet`

### Step 3.4 — `src/transform/validate.py`
- pandera DataFrameSchema enforcing:
  - `iso3_code`: string, 3 chars
  - `year`: int, 1990–2025
  - `diabetes_prev_agestd`: float, 0–50 (nullable)
  - `mean_bmi`: float, 15–45 (nullable)
  - Other indicators with reasonable ranges
- Log validation failures, don't crash — report coverage stats

**Verification:** Run `make transform`. Check `data/processed/country_year_panel.parquet` exists. Validate 150+ unique countries, 20+ years of data.

---

## Phase 4: Load
**Goal:** Load the processed panel into SQLite.

### Step 4.1 — `src/load/to_sqlite.py`
- Read `data/processed/country_year_panel.parquet`
- `df.to_sql('country_year_indicators', con, if_exists='replace')` — idempotent full reload
- Also ensure `dim_country` and `dim_indicator` are populated
- Print summary: row count, country count, year range

**Verification:** `make load`. Query `SELECT COUNT(*), COUNT(DISTINCT iso3_code), MIN(year), MAX(year) FROM country_year_indicators`.

---

## Phase 5: Feature Engineering
**Goal:** Build the model-ready feature matrix with lags, interactions, and encodings.

### Step 5.1 — `src/features/engineer.py`
- Read from `country_year_indicators` (SQLite or parquet)
- **Lag features:** `bmi_lag_5`, `bmi_lag_10`, `glucose_lag_5`, `overweight_lag_5` — using `groupby('iso3_code').shift(N)`
- **Rate of change:** `bmi_5yr_change`, `urban_pop_change_10yr`
- **Interactions:** `glucose_bmi_interaction`, `inactivity_obesity_interaction`
- **Categorical:** `income_group` → ordinal (Low=1..High=4), `region` → one-hot
- **Derived:** `health_exp_relative = health_exp_per_capita / gdp_per_capita_ppp`
- Drop rows where target (`diabetes_prev_agestd`) is null
- Drop rows with insufficient feature coverage (require ≥5 non-null features)
- Output: `data/model_ready/global_features.parquet`

**Verification:** `make features`. Check parquet has expected columns, reasonable row count (likely 2000–4000 country-year obs).

---

## Phase 6: Models
**Goal:** Train OLS → Ridge → RF → XGBoost, evaluate with GroupKFold and temporal holdout.

### Step 6.1 — `src/models/evaluate.py`
- Shared evaluation utilities:
  - `evaluate_model(model, X_train, X_test, y_train, y_test)` → dict of RMSE, MAE, R², MAPE
  - `group_kfold_cv(model, X, y, groups, n_splits=5)` → cross-val metrics
  - `temporal_split(df, holdout_years=[2020, 2021, 2022])` → train/test DataFrames
  - Print/save results to `outputs/reports/model_comparison.json`

### Step 6.2 — `src/models/baseline.py`
- Load `data/model_ready/global_features.parquet`
- Define feature columns (all except target, iso3_code, year, region dummies stay)
- Train/eval Pooled OLS (`LinearRegression`)
- Train/eval Ridge (`Ridge` with alpha CV)
- Both with GroupKFold by country + temporal holdout
- Save models with `joblib.dump()` to `outputs/`
- Print metrics comparison table

### Step 6.3 — `src/models/advanced.py`
- Train/eval Random Forest (`RandomForestRegressor`, n_estimators=200)
- Train/eval XGBoost (`XGBRegressor`, with basic hyperparameter tuning)
- Same evaluation framework as baseline
- Save best model (expected: XGBoost)
- Print full 4-model comparison table

**Verification:** `make train`. All 4 models produce metrics. XGBoost RMSE < OLS RMSE. R² > 0.7 target.

---

## Phase 7: Simulations
**Goal:** Project features to 2050, run 5 what-if scenarios, store results.

### Step 7.1 — `src/simulation/scenarios.py`
- Define scenario dataclass/dict:
  ```python
  SCENARIOS = {
      "reduce_obesity_20pct": {"overweight_prev_pct": 0.80, "mean_bmi": 0.97},
      "increase_activity": {"physical_inactivity_pct": 0.70},
      "double_health_exp": {"health_exp_per_capita": 2.0},
      "combined_intervention": {"overweight_prev_pct": 0.80, "mean_bmi": 0.97, "physical_inactivity_pct": 0.70, "health_exp_per_capita": 2.0},
      "worst_case_obesity": {"overweight_prev_pct": 1.30, "mean_bmi": 1.04},
  }
  ```

### Step 7.2 — `src/simulation/engine.py`
- Load best model (XGBoost) and feature matrix
- For each country (focus on top 10: USA, IND, CHN, BRA, MEX, IDN, DEU, GBR, JPN, NGA):
  1. Fit linear trend per feature on last 10 years
  2. Extrapolate to target years: 2030, 2040, 2050
  3. Predict baseline: `model.predict(projected_features)`
  4. For each scenario: multiply specified features by multipliers, re-predict
  5. Compute delta = scenario − baseline
  6. Clamp all predictions to [0, 50]
- Insert results into `simulation_results` SQLite table
- Save summary to `outputs/reports/simulation_results.csv`

**Verification:** `make simulate`. Check `simulation_results` table has 5 scenarios × 10 countries × 3 years = 150 rows. Deltas are negative for intervention scenarios.

---

## Phase 8: Explainability
**Goal:** SHAP values, permutation importance, and feature importance visualizations.

### Step 8.1 — `src/explainability/shap_analysis.py`
- Load trained XGBoost model and test set
- **Permutation importance:** `sklearn.inspection.permutation_importance` → bar chart saved to `outputs/figures/permutation_importance.png`
- **SHAP TreeExplainer:**
  - Compute SHAP values for test set
  - Beeswarm summary plot → `outputs/figures/shap_beeswarm.png`
  - Dependence plots for top 3 features → `outputs/figures/shap_dep_{feature}.png`
  - Force plots for USA, IND, BRA (most recent year) → `outputs/figures/shap_force_{country}.png`
- Print written interpretation of top findings

**Verification:** All PNG files generated in `outputs/figures/`. SHAP values sum to (prediction − base_value) for each row.

---

## Phase 9: Notebooks (MVP wrap-up)
**Goal:** 4 narrative notebooks that showcase the pipeline results.

### Step 9.1 — `notebooks/01_eda_global.ipynb`
- Load `country_year_indicators`, show coverage heatmap (countries × years)
- Distribution plots for key indicators
- Time series of diabetes prevalence for top 10 countries
- Correlation matrix of all indicators

### Step 9.2 — `notebooks/02_modeling.ipynb`
- Load model comparison results
- Actual vs predicted scatter plots
- Residual analysis
- Feature importance comparison across models

### Step 9.3 — `notebooks/03_simulations.ipynb`
- Load simulation results
- Bar charts: baseline vs scenario for each country
- Grouped comparison across scenarios
- Global impact summary

### Step 9.4 — `notebooks/04_explainability.ipynb`
- Load and display SHAP plots
- Written interpretation of findings
- Per-country deep dives (USA, India, Brazil)

---

## Phase 10: README + Final Polish
- `README.md` with project overview, setup instructions, architecture diagram, key findings
- Ensure `make all` runs end-to-end from clean state

---

## Implementation Sessions (how to ask Claude Code to build this)

| Session | Phases | What to ask Claude Code |
|---|---|---|
| **1** | Phase 1 | "Implement Phase 1 from EXECUTION_PLAN.md — project scaffolding" |
| **2** | Phase 2 | "Implement Phase 2 — extraction layer (WHO, World Bank, OWID)" |
| **3** | Phase 3–4 | "Implement Phases 3–4 — transform, validate, and load pipeline" |
| **4** | Phase 5–6 | "Implement Phases 5–6 — feature engineering and model training" |
| **5** | Phase 7–8 | "Implement Phases 7–8 — simulations and explainability" |
| **6** | Phase 9–10 | "Implement Phases 9–10 — notebooks and README" |

---

## Verification Checklist (End-to-End)
1. `make all` completes without errors
2. `country_year_indicators` has 150+ countries, 20+ years
3. pandera validation passes
4. XGBoost test RMSE < OLS baseline RMSE; R² > 0.7
5. 150 simulation result rows (5 scenarios × 10 countries × 3 years)
6. SHAP plots generated in `outputs/figures/`
7. All notebooks run without errors
