# Phase 10: Fix Models, Visualize Results, Automate Pipeline

## Context

The SVM model and simulation engine have critical bugs producing **inverted and converging results**. Before any visualization or automation work, these must be fixed. Once fixed, we build visualizations for SVM, Belief Network, and simulations, then wire everything into a single automation script.

### Problems Discovered

| # | Problem | Where | Impact |
|---|---------|-------|--------|
| 1 | Column name typo `bmi_5y_change` vs `bmi_5yr_change` | `engine.py:58` | Derived features never recalculated after scenario multipliers |
| 2 | `fillna(0)` before scaling | `engine.py:64` | Pushes features far out-of-distribution, SVR predicts near mean |
| 3 | Default SVR params (C=1.0) | `svm.py:30` | 97% support vectors = underfitting, poor sensitivity |
| 4 | No extrapolation clamping | `engine.py:18-33` | RBF kernel collapses to mean for distant points (convergence to ~14.8% by 2050) |
| 5 | 50% feature coverage threshold | `feature_matrix.py:54` | USA, MEX, GBR, NGA dropped entirely from analysis |
| 6 | Inverted results | Combination of 1-4 | India: reducing BMI *increases* prevalence; worst-case *decreases* it |

---

## Part 1: Fix SVM Model and Simulation Engine

### Step 1.1 -- Fix column name typo
**File:** `src/simulations/engine.py` line 58  
**Change:** `bmi_5y_change` --> `bmi_5yr_change`  
**Why:** Without this fix, the `bmi_5yr_change` feature is never recalculated after applying BMI scenario multipliers, so the model sees inconsistent feature values.

### Step 1.2 -- Add hyperparameter tuning to SVM
**File:** `src/models/svm.py`  
**What:**
- Replace `SVR(kernel="rbf")` with `GridSearchCV` (or `RandomizedSearchCV` if too slow)
- Parameter grid:
  - `C`: [0.1, 1, 10, 100, 1000]
  - `epsilon`: [0.01, 0.05, 0.1, 0.2]
  - `gamma`: ['scale', 0.001, 0.01, 0.1]
- Use `GroupKFold(n_splits=5)` grouped by `iso3_code` for cross-validation
- Save best model, print best params and support vector count
- Save training medians: `joblib.dump(X_train.median(), "outputs/models/svm_training_medians.pkl")`
- Save feature bounds (min/max per feature): `joblib.dump(feature_bounds, "outputs/models/svm_feature_bounds.pkl")`

**Why:** Default C=1.0 produces 97% support vectors (underfitting). A higher C will produce a more flexible decision boundary with better sensitivity to feature changes.

### Step 1.3 -- Fix fillna(0) in simulation engine
**File:** `src/simulations/engine.py` line 64  
**Change:** Replace `fillna(0)` with `fillna(training_medians)` where medians are loaded from `outputs/models/svm_training_medians.pkl`  
**Why:** Filling with 0 produces extreme scaled values after StandardScaler (e.g., health_exp_per_capita mean ~244, filling with 0 creates a scaled value of -0.4 to -2+). Training medians are neutral and keep predictions in-distribution.

### Step 1.4 -- Add extrapolation clamping
**File:** `src/simulations/engine.py` in `extrapolate_country()` function  
**What:** After computing projected feature values, clamp each to its training min/max range (loaded from `outputs/models/svm_feature_bounds.pkl`)  
**Why:** The RBF kernel formula `K(x,x') = exp(-gamma * ||x-x'||^2)` approaches zero for distant points, causing all predictions to collapse toward the training mean (~14.8%). Clamping keeps projections within the model's reliable range.

### Step 1.5 -- Fix missing countries
**File:** `src/features/feature_matrix.py` line ~54  
**Diagnosis steps:**
1. Query `raw_worldbank` for USA/MEX/GBR/NGA to confirm data exists in the database
2. Check if ISO3 codes match between WHO and World Bank sources in the merge
3. If merge works but row coverage is too low: change the 50% threshold

**Proposed fix:** Replace the single 50% threshold with a two-tier rule:
- Keep rows where core WHO features (mean_bmi, raised_blood_glucose_pct, physical_inactivity_pct) are all present, OR
- Keep rows where overall feature coverage >= 30%

### Step 1.6 -- Verify all fixes
- Re-run: `python -m src.models.svm && python -m src.simulations.engine`
- **Check 1:** India reduce_bmi_5pct shows LOWER prevalence than baseline
- **Check 2:** 2050 predictions differ meaningfully from 2040 (not converging)
- **Check 3:** USA, MEX, GBR, NGA appear in simulation results
- **Check 4:** Support vector count is well below 97% of training data

---

## Part 2: SVM Visualizations

**New file:** `src/explainability/svm_visualizations.py`  
**Output directory:** `outputs/figures/`

### Plot 2.1 -- Actual vs Predicted Scatter
- **Data:** Test set (2014-2016 holdout), y_true vs y_pred
- **Plot:** Scatter plot with 45-degree reference line, annotated with R2 and RMSE
- **Library:** matplotlib + seaborn
- **Output:** `outputs/figures/svm_actual_vs_predicted.png`
- **Purpose:** Shows whether the model is systematically over- or under-predicting, and at which prevalence ranges

### Plot 2.2 -- Feature Importance (Permutation)
- **Data:** Test set, `sklearn.inspection.permutation_importance(model, X_test_scaled, y_test, n_repeats=30)`
- **Plot:** Horizontal bar chart, top 15 features, with error bars (std across repeats)
- **Library:** matplotlib
- **Output:** `outputs/figures/svm_feature_importance.png`
- **Purpose:** Shows which indicators the SVM considers most important for predicting diabetes prevalence. This directly feeds into the BN and automation pipeline.

### Plot 2.3 -- SHAP Beeswarm
- **Data:** Subsample of test set (100 rows for speed)
- **Plot:** SHAP beeswarm summary plot
- **Library:** shap (`KernelExplainer` with 100 background samples)
- **Output:** `outputs/figures/svm_shap_summary.png`
- **Purpose:** Shows not just importance but *direction* of each feature's effect. E.g., does high BMI push predictions up or down?
- **Note:** Computationally expensive (~5-10 min). Can be skipped initially if needed.

### Plot 2.4 -- Residual Analysis
- **Data:** Test set residuals (y_true - y_pred)
- **Plot:** Two subplots: (a) residuals vs predicted value, (b) residual histogram/KDE
- **Library:** matplotlib + seaborn
- **Output:** `outputs/figures/svm_residual_analysis.png`
- **Purpose:** Check for heteroscedasticity (residuals growing with prediction) and systematic bias

### Plot 2.5 -- Partial Dependence Plots
- **Data:** Training data (subsample to 500 rows for speed), top 3 features
- **Plot:** PDP with ICE curves
- **Library:** `sklearn.inspection.PartialDependenceDisplay`
- **Output:** `outputs/figures/svm_partial_dependence.png`
- **Purpose:** Shows the marginal effect of each feature on prediction, holding others constant

---

## Part 3: Belief Network Visualizations

**New file:** `src/explainability/bn_visualizations.py`

### Plot 3.1 -- Network Graph with Edge Strengths
- **Data:** BN model structure (8 nodes, 9 edges) loaded from `outputs/models/bn_model.pkl`
- **Plot:** Directed graph with node labels, edge width proportional to mutual information between connected nodes
- **Library:** networkx + matplotlib
- **Output:** `outputs/figures/bn_network_graph.png`
- **Purpose:** Visual overview of what the BN considers to be the causal structure of diabetes prevalence

### Plot 3.2 -- Conditional Probability Heatmaps
- **Data:** CPD tables from the trained BN model
- **Plot:** Heatmap for the `diabetes_prev_agestd` node
- Since 5 parents create 3^5=243 combinations: fix 3 parents at "medium", show 3x3 heatmap for the 2 most important parents (likely glucose + BMI)
- **Library:** seaborn heatmap
- **Output:** `outputs/figures/bn_cpd_diabetes.png`
- **Purpose:** Shows the BN's learned conditional probabilities -- what combinations of parent states lead to high/medium/low diabetes

### Plot 3.3 -- Causal Query Bar Chart
- **Data:** Query the BN with different evidence settings using VariableElimination
- **Queries to run:**
  1. P(diabetes | mean_bmi=high) vs P(diabetes | mean_bmi=low)
  2. P(diabetes | glucose=high) vs P(diabetes | glucose=low)
  3. P(diabetes | inactivity=high) vs P(diabetes | inactivity=low)
  4. P(diabetes | health_exp=high) vs P(diabetes | health_exp=low)
  5. P(diabetes | bmi=high, glucose=high) vs P(diabetes | bmi=low, glucose=low)
- **Plot:** Grouped bar chart showing full probability distributions (low/medium/high)
- **Library:** matplotlib
- **Output:** `outputs/figures/bn_causal_queries.png`
- **Purpose:** Directly answers "what does the BN consider strong relationships?" by showing how changing evidence shifts the diabetes probability distribution

---

## Part 4: Simulation Visualizations

**New file:** `src/explainability/simulation_visualizations.py`  
**Prerequisite:** Part 1 fixes must be verified before generating these.

### Plot 4.1 -- Scenario Comparison Per Country
- **Data:** `outputs/reports/simulation_results.csv` (re-generated after fixes)
- **Plot:** Grouped bar chart faceted by country, one bar per scenario at year 2050
- **Library:** matplotlib or plotly
- **Output:** `outputs/figures/sim_scenario_comparison.png`

### Plot 4.2 -- Timeline Projections
- **Data:** Simulation results + historical data from feature matrix
- **Plot:** Line chart per country: year on x-axis, prevalence on y-axis, one line per scenario
- Include historical data points, vertical dashed line at last observed year
- **Library:** matplotlib
- **Output:** `outputs/figures/sim_timeline_{country}.png` (one per country)

### Plot 4.3 -- Cross-Country Heatmap
- **Data:** Simulation results for baseline and combined_intervention scenarios
- **Plot:** Heatmap with countries on y-axis, years on x-axis, color = prevalence
- **Library:** seaborn
- **Output:** `outputs/figures/sim_cross_country_heatmap.png`

### Plot 4.4 -- Intervention Impact Delta Chart
- **Data:** (scenario_prevalence - baseline_prevalence) for each country/scenario at 2050
- **Plot:** Diverging horizontal bar chart (negative = prevalence reduction = good, positive = increase = bad)
- **Library:** matplotlib
- **Output:** `outputs/figures/sim_intervention_deltas.png`
- **Purpose:** The clearest way to see "how much does each intervention actually help?"

---

## Part 5: Full Pipeline Automation Script

**New file:** `src/pipeline/run_full_pipeline.py`  
**Run with:** `python -m src.pipeline.run_full_pipeline`

### Pipeline Architecture

```
run_full_pipeline.py
|
+-- Step 1: Train SVM (with hyperparameter tuning)
|   +-- Load feature matrix
|   +-- Temporal split
|   +-- GridSearchCV for SVR
|   +-- Save model, scaler, medians, bounds
|   +-- Extract permutation importance --> top_features list
|
+-- Step 2: Train BN on top features from SVM
|   +-- Select top N features from permutation importance
|   +-- Define BN structure using domain knowledge + selected features
|   +-- Train BN with MaximumLikelihoodEstimator
|   +-- Run causal queries to identify strongest relationships
|   +-- Auto-generate scenarios from BN results
|       (e.g., if P(diabetes|glucose=low) << P(diabetes|glucose=high),
|        create scenario reducing glucose by the amount needed to move
|        from "high" to "medium" bin edge)
|
+-- Step 3: Run simulations
|   +-- Use SVM model for predictions
|   +-- Use both predefined + BN-generated scenarios
|   +-- Extrapolate features with clamping
|   +-- Apply scenario multipliers, predict
|   +-- Save results to CSV and SQLite
|
+-- Step 4: Generate all visualizations
|   +-- SVM: actual vs pred, importance, SHAP, residuals, PDP
|   +-- BN: network graph, CPDs, causal queries
|   +-- Simulation: scenarios, timelines, cross-country, deltas
|
+-- Step 5: Generate summary report
    +-- Model metrics (R2, RMSE, MAE, MAPE)
    +-- Top 10 features by importance
    +-- Strongest BN relationships
    +-- Scenario impact summary table
    +-- Save to outputs/reports/pipeline_report.txt
```

### Key Design Decisions

1. **BN-driven scenario generation:** The pipeline queries the BN to find which evidence settings cause the largest shift in P(diabetes=high). These are converted into numerical scenario multipliers using the bin edges. This makes the system "unsupervised" -- the BN discovers impactful interventions rather than us defining them manually.

2. **Configuration block at the top** for easy parameter adjustment (paths, holdout years, number of top features, projection years, etc.)

3. **Each step wrapped in try/except** -- reports which step failed and why, but continues with remaining steps where possible.

4. **Importable functions:** Each step is a function so individual parts can be called independently.

---

## File Manifest

### Files to Modify
| File | Changes |
|------|---------|
| `src/models/svm.py` | GridSearchCV, save medians/bounds, permutation importance |
| `src/simulations/engine.py` | Fix typo, fix fillna, add clamping, load medians/bounds |
| `src/features/feature_matrix.py` | Fix country dropout threshold |

### Files to Create
| File | Purpose |
|------|---------|
| `src/explainability/__init__.py` | Package init |
| `src/explainability/svm_visualizations.py` | 5 SVM visualization functions |
| `src/explainability/bn_visualizations.py` | 3 BN visualization functions |
| `src/explainability/simulation_visualizations.py` | 4 simulation visualization functions |
| `src/pipeline/__init__.py` | Package init |
| `src/pipeline/run_full_pipeline.py` | Full automation script |

---

## Implementation Order

| Session | What | Depends On |
|---------|------|------------|
| **A** | Part 1: Fix SVM + simulation engine | Nothing |
| **B** | Part 2: SVM visualizations | Part 1 verified |
| **C** | Part 3: BN visualizations | Nothing (can parallel with B) |
| **D** | Part 4: Simulation visualizations | Part 1 verified |
| **E** | Part 5: Full pipeline automation | Parts 1-4 complete |

---

## Suggestions and Notes

1. **After tuning, consider feature selection.** With 27 features (many correlated lags), dropping features with zero or negative permutation importance will make the RBF kernel more effective and reduce noise.

2. **SHAP is optional for first pass.** `KernelExplainer` on SVR is slow. Permutation importance gives 80% of the insight in 1% of the time. Add SHAP once everything else works.

3. **The convergence problem may not fully disappear with clamping alone.** If extrapolated features cluster tightly (because many features hit their training max), predictions will still be similar. This is actually correct behavior -- the model is saying "I can't reliably distinguish these scenarios at such extreme projections." Frame 2050 results as "conservative estimates" in the report.

4. **For the BN, consider re-training after Part 1 fixes** -- the feature matrix may change if the coverage threshold is adjusted, which would affect the discretization bin edges.

5. **World Bank data investigation for missing countries** should be done first (Step 1.5) since it may require re-running the extraction and transform pipeline before model retraining.
