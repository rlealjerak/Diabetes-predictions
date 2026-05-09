# Jupyter Notebook Plan: Diabetes Pipeline Results

## File
`notebooks/results.ipynb`

## Goal
Display all pipeline outputs in one place — no retraining, no recomputing.
Everything loads from saved files in `outputs/`.

---

## Cell Structure

### Cell 1 — Markdown: Title
```
# Diabetes Prevalence Analysis
## Pipeline Results
Brief description of the project: SVM + Belief Network models, simulations across 10 countries and 7 scenarios.
```

---

### Cell 2 — Code: Imports
```python
import joblib, pickle, pandas as pd
from IPython.display import Image, display
from pathlib import Path
```

---

### Section 1: SVM Model

### Cell 3 — Markdown
```
## SVM Model
Trained with HalvingGridSearchCV on historical WHO + World Bank data.
Test set: 2014–2016 holdout.
```

### Cell 4 — Code: Print SVM metrics
Load `outputs/models/svm_permutation_importance.pkl` and print best metrics.

### Cell 5 — Code: Display SVM plots
Display the 4 saved PNGs:
- `outputs/figures/svm_actual_vs_predicted.png`
- `outputs/figures/svm_feature_importance.png`
- `outputs/figures/svm_residual_analysis.png`
- `outputs/figures/svm_partial_dependence.png`

---

### Section 2: Belief Network

### Cell 6 — Markdown
```
## Belief Network
Trained on discretized features (low/medium/high bins).
Shows causal structure and conditional probabilities.
```

### Cell 7 — Code: Print causal query results
Load `outputs/models/bn_causal_queries.pkl` and print the delta table
(P(diabetes=high | feature=high) - P(diabetes=high | feature=low)).

### Cell 8 — Code: Display BN plots
Display the 2 saved PNGs:
- `outputs/figures/bn_network_graph.png`
- `outputs/figures/bn_cpd_diabetes.png`

---

### Section 3: Simulations

### Cell 9 — Markdown
```
## Simulations
7 scenarios projected to 2030, 2040, 2050 across 10 countries.
```

### Cell 10 — Code: Show simulation table
Load `outputs/reports/simulation_results.csv`, pivot and display the 2050 summary table.

### Cell 11 — Code: Display simulation plots
Display the 4 saved PNGs:
- `outputs/figures/sim_scenario_comparison.png`
- `outputs/figures/sim_cross_country_heatmap.png`
- `outputs/figures/sim_intervention_deltas.png`
- A few selected `sim_timeline_{country}.png` (e.g. IND, USA, CHN)

---

### Section 4: Summary Report

### Cell 12 — Markdown
```
## Pipeline Report
```

### Cell 13 — Code: Print report
```python
print(open("outputs/reports/pipeline_report.txt").read())
```

---

## Notes
- Create `notebooks/` folder before creating the file
- Run cells top to bottom — each cell is independent
- If a figure is missing, re-run `python -m src.automation.run_pipeline` first
