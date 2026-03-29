# Global Diabetes Futures — Execution Plan (Revised)

## Context
We have a fully designed project blueprint (see BLUEPRINT.md) for an end-to-end public health data pipeline. This revised plan changes the approach: instead of building all extractors first, we start with **one data source (WHO GHO)**, build the **full pipeline skeleton end-to-end**, then iterate by adding more sources once the process is proven.

**Key changes from original plan:**
- Single source first (WHO GHO), expand later
- Models: **SVM** and **Bayesian Belief Network** (replacing OLS → Ridge → RF → XGBoost)
- Iterative approach: get results first, then add data and automation

---

## Phase 1: Project Scaffolding ✅ DONE
**Goal:** Repo structure, dependencies, git init, database schema.

- Folder structure created
- `pyproject.toml` with dependencies
- `.gitignore`, `Makefile`, `BLUEPRINT.md`
- SQLite database initialized

---

## Phase 2: WHO GHO Extraction (IN PROGRESS)
**Goal:** Complete the WHO extractor — the sole data source for the first full pipeline pass.

### Step 2.1 — Finish `src/extract/who_gho.py`
- 5 of 6 indicators already working: `NCD_DIABETES_PREVALENCE_AGESTD`, `NCD_BMI_MEAN`, `NCD_GLUC_04`, `NCD_PAA`, `NCDMORT3070`
- Investigate `NCD_GLUC_01` (mean fasting blood glucose) — returns no data with `Dim1 eq 'SEX_BTSX'` filter. Need to check available Dim1 values for this indicator.
- Confirm raw JSON saved to `data/raw/who/` and rows inserted into `raw_who_gho`

### Step 2.2 — Database schema (`db/schema.sql`)
- Create schema file with `raw_who_gho`, `dim_country`, `dim_indicator`, `country_year_indicators`, `simulation_results`
- Include `raw_worldbank` and `raw_owid` table DDL for future use, but they won't be populated yet

**Verification:** `python -m src.extract.who_gho` runs successfully. `SELECT COUNT(*) FROM raw_who_gho` returns thousands of rows.

---

## Phase 3: Transform + Clean (WHO only)
**Goal:** Clean and prepare WHO data into a unified table ready for feature engineering.

### Step 3.1 — `src/transform/clean.py`
- Read from `raw_who_gho` SQLite table
- Extract `SpatialDim` (ISO3), `TimeDim` (year), `NumericValue`
- Pivot indicators into columns: `diabetes_prev_agestd`, `mean_bmi`, `mean_fasting_glucose`, `raised_blood_glucose_pct`, `physical_inactivity_pct`, `ncd_mortality_prob`
- Flag outliers using IQR per indicator (log, don't remove)
- Output: cleaned DataFrame saved to `data/processed/who_clean.parquet`

### Step 3.2 — `src/transform/normalize.py`
- WHO already uses ISO3 in `SpatialDim` — minimal normalization needed
- Standardize column names to match `country_year_indicators` schema
- Filter to both-sexes records (already filtered at extraction)
- Populate `dim_country` from WHO country codes (use `pycountry` for region/income_group)

### Step 3.3 — `src/transform/merge.py`
- With single source, this is simpler: build `(iso3_code, year)` panel from WHO data
- Linear interpolation for gaps of 1-2 years within each country
- Socioeconomic columns (`gdp_per_capita_ppp`, `urban_pop_pct`, etc.) will be NULL until World Bank is added — that's fine
- Output: `data/processed/country_year_panel.parquet`

### Step 3.4 — `src/transform/validate.py`
- pandera schema enforcing reasonable ranges:
  - `diabetes_prev_agestd`: 0–50
  - `mean_bmi`: 15–45
  - `physical_inactivity_pct`: 0–100
  - `year`: 1990–2025
- Log validation failures and coverage stats

**Verification:** `data/processed/country_year_panel.parquet` has 150+ countries, 20+ years. Pandera validation passes.

---

## Phase 4: Load to SQLite
**Goal:** Load processed panel into the database.

### Step 4.1 — `src/load/to_sqlite.py`
- Read `data/processed/country_year_panel.parquet`
- `df.to_sql('country_year_indicators', con, if_exists='replace')`
- Populate `dim_country` and `dim_indicator`
- Print summary: row count, country count, year range

**Verification:** Query `SELECT COUNT(*), COUNT(DISTINCT iso3_code), MIN(year), MAX(year) FROM country_year_indicators`.

---

## Phase 5: Feature Engineering (WHO features only)
**Goal:** Build model-ready feature matrix from the available WHO indicators.

### Step 5.1 — `src/features/engineer.py`
- Read from `country_year_indicators`
- **Available features (WHO only):** mean_bmi, mean_fasting_glucose, raised_blood_glucose_pct, physical_inactivity_pct, ncd_mortality_prob
- **Lag features:** `bmi_lag_5`, `bmi_lag_10`, `glucose_lag_5` — using `groupby('iso3_code').shift(N)`
- **Rate of change:** `bmi_5yr_change`
- **Interactions:** `glucose_bmi_interaction`, `inactivity_bmi_interaction`
- Drop rows where target (`diabetes_prev_agestd`) is null
- Drop rows with insufficient feature coverage
- Output: `data/model_ready/global_features.parquet`

**Note:** Feature set will expand when World Bank indicators are added in Phase 8. The pipeline is designed to accommodate this without restructuring.

**Verification:** Parquet file has expected columns. Reasonable row count (likely 2000+ country-year observations).

---

## Phase 6: Models — SVM and Bayesian Belief Network
**Goal:** Train and evaluate two model types on the WHO-only feature set.

### Step 6.1 — `src/models/evaluate.py`
- Shared evaluation utilities:
  - `evaluate_model(model, X_train, X_test, y_train, y_test)` → dict of RMSE, MAE, R², MAPE
  - `group_kfold_cv(model, X, y, groups, n_splits=5)` → cross-val metrics
  - `temporal_split(df, holdout_years=[2020, 2021, 2022])` → train/test DataFrames
  - Save results to `outputs/reports/model_comparison.json`

### Step 6.2 — `src/models/svm.py`
- Support Vector Machine (regression: `SVR` from scikit-learn)
- Feature scaling required (StandardScaler pipeline)
- Kernel selection: test RBF and linear, pick best via CV
- Evaluate with GroupKFold by country + temporal holdout
- Save model with `joblib.dump()`

### Step 6.3 — `src/models/belief_network.py`
- Bayesian Belief Network (using `pgmpy` or similar library)
- Define network structure based on domain knowledge: BMI → glucose → diabetes, inactivity → BMI → diabetes, etc.
- Parameter learning from data
- Evaluate probabilistic predictions against actuals
- Save model

**Verification:** Both models produce metrics. Compare SVM vs BN performance. Document which performs better and why.

---

## Phase 7: Simulations
**Goal:** Project features to 2050, run what-if scenarios with trained models.

### Step 7.1 — `src/simulation/scenarios.py`
- Define scenarios (adjusted for WHO-only features):
  - **Reduce BMI 5%:** `mean_bmi *= 0.95`
  - **Reduce blood glucose 10%:** `raised_blood_glucose_pct *= 0.90`
  - **Increase physical activity:** `physical_inactivity_pct *= 0.70`
  - **Combined intervention:** all above
  - **Worst-case:** `mean_bmi *= 1.05`, `physical_inactivity_pct *= 1.20`

### Step 7.2 — `src/simulation/engine.py`
- Load best model and feature matrix
- For top 10 countries (USA, IND, CHN, BRA, MEX, IDN, DEU, GBR, JPN, NGA):
  1. Fit linear trend per feature on last 10 years
  2. Extrapolate to 2030, 2040, 2050
  3. Predict baseline
  4. Apply scenario multipliers, re-predict
  5. Compute deltas, clamp to [0, 50]
- Insert into `simulation_results` table
- Save CSV to `outputs/reports/`

**Verification:** `simulation_results` table populated. Intervention scenarios show negative deltas (reduced diabetes).

---

## Phase 8: Add World Bank Data (Iteration 2)
**Goal:** Expand the pipeline with socioeconomic context from World Bank.

### Step 8.1 — `src/extract/worldbank.py`
- Indicators: `SH.STA.OWAD.ZS`, `NY.GDP.PCAP.PP.CD`, `SP.URB.TOTL.IN.ZS`, `SH.XPD.CHEX.PC.CD`, `SH.XPD.CHEX.GD.ZS`, `SP.POP.65UP.TO.ZS`, `SP.DYN.LE00.IN`, `SP.POP.TOTL`
- API extraction, save to `data/raw/worldbank/`, insert into `raw_worldbank`

### Step 8.2 — Update transform pipeline
- `clean.py`: add World Bank cleaning (drop aggregates, coerce types)
- `normalize.py`: World Bank already uses ISO3
- `merge.py`: left-join World Bank indicators onto the existing panel
- `validate.py`: add ranges for new indicators

### Step 8.3 — Update feature engineering
- Add new features: `gdp_per_capita_ppp`, `urban_pop_pct`, `health_exp_per_capita`, etc.
- Add derived features: `health_exp_relative`, `urban_pop_change_10yr`
- Add categorical: `income_group` ordinal encoding

### Step 8.4 — Retrain models
- Retrain SVM and BN with expanded feature set
- Compare metrics: did socioeconomic features improve predictions?
- Update simulation scenarios (can now include `health_exp_per_capita *= 2.0`)

**Verification:** Full `make all` runs with both sources. Model metrics improve with additional features.

---

## Phase 9: Add OWID Data + Further Sources (Iteration 3)
**Goal:** Fill gaps with OWID, add any other valuable sources.

### Step 9.1 — `src/extract/owid_csv.py`
- Download and parse OWID CSVs for BMI/obesity trends
- Insert into `raw_owid`

### Step 9.2 — Update pipeline
- Integrate OWID into transform/merge
- Retrain models, update simulations

---

## Phase 10: Explainability, Notebooks, Polish
**Goal:** Interpretation, documentation, and project wrap-up.

### Step 10.1 — `src/explainability/shap_analysis.py`
- Permutation importance, SHAP values (adapt for SVM/BN), partial dependence plots
- Save figures to `outputs/figures/`

### Step 10.2 — Notebooks
- `01_eda_global.ipynb` — Coverage, distributions, time series, correlations
- `02_modeling.ipynb` — Model comparison, actual vs predicted, residuals
- `03_simulations.ipynb` — Scenario comparisons, global impact
- `04_explainability.ipynb` — SHAP plots, interpretation

### Step 10.3 — README + automation
- `README.md` with setup, architecture, findings
- `make all` runs end-to-end from clean state
- Automation improvements

---

## Implementation Sessions

| Session | Phases | Focus |
|---|---|---|
| **1** | Phase 1 | Project scaffolding ✅ |
| **2** | Phase 2 | WHO extraction (in progress) |
| **3** | Phase 3–4 | Transform + load (WHO only) |
| **4** | Phase 5–6 | Features + SVM/BN models |
| **5** | Phase 7 | Simulations |
| **6** | Phase 8 | Add World Bank, retrain |
| **7** | Phase 9 | Add OWID, retrain |
| **8** | Phase 10 | Explainability, notebooks, polish |

---

## End-to-End Verification Checklist
1. `make all` completes without errors (WHO-only pass)
2. `country_year_indicators` has 150+ countries, 20+ years
3. pandera validation passes
4. SVM and BN models produce valid metrics
5. Simulation results table populated with scenario deltas
6. After Phase 8: metrics improve with World Bank features
7. SHAP/explainability plots generated
8. All notebooks run without errors
