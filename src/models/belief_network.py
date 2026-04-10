import pandas as pd 
from src.models.evaluate import temporal_split, evaluate_model  
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
import pickle 

df = pd.read_parquet("data/model_ready/global_features.parquet")
train, test = temporal_split(df)

# Discretize data 
columns_to_use = ["mean_bmi", "raised_blood_glucose_pct", "physical_inactivity_pct", "ncd_mortality_prob", "diabetes_prev_agestd", "health_exp_per_capita", "gdp_per_capita_ppp", "urban_pop_pct"]

# Drop NaN rows before discretizing data 
train_clean = train[columns_to_use].dropna()
test_clean = test[columns_to_use].dropna()

train_disc = train_clean.copy()
test_disc = test_clean.copy()

bin_edges = {}
for col in columns_to_use:
    train_disc[col], bins = pd.qcut(train_clean[col], q=3, labels=["low", "medium", "high"], retbins=True)
    bin_edges[col] = bins.copy()  # save original edges for midpoints
    cut_bins = bins.copy()
    cut_bins[0] = -float("inf")  # extend edges so no test value falls outside
    cut_bins[-1] = float("inf")
    test_disc[col] = pd.cut(test_clean[col], bins=cut_bins, labels=["low", "medium", "high"])

print(f"Train rows after discretize: {len(train_disc.dropna())}")
print(f"Test rows after discretize: {len(test_disc.dropna())}")

# Define the network structure 
model = DiscreteBayesianNetwork([
    ("mean_bmi", "raised_blood_glucose_pct"),
    ("mean_bmi", "diabetes_prev_agestd"),
    ("raised_blood_glucose_pct", "diabetes_prev_agestd"),
    ("physical_inactivity_pct", "mean_bmi"),
    ("physical_inactivity_pct", "diabetes_prev_agestd"),
    ("ncd_mortality_prob", "diabetes_prev_agestd"),
    ("gdp_per_capita_ppp", "health_exp_per_capita"),
    ("health_exp_per_capita", "diabetes_prev_agestd"),
    ("urban_pop_pct", "physical_inactivity_pct")
])

# Fit the model to the training data 
model.fit(train_disc.dropna(), estimator=MaximumLikelihoodEstimator)

# Evaluate model on test data
inference = VariableElimination(model)

predictions = [] 
for _, row in test_disc.dropna().iterrows():
    evidence = {col: row[col] for col in columns_to_use if col != "diabetes_prev_agestd"}
    pred = inference.map_query(variables=["diabetes_prev_agestd"], evidence=evidence)
    predictions.append(pred["diabetes_prev_agestd"])

# Convert predictions back to numeric values for evaluation
midpoints = {} 
for col, edges in bin_edges.items(): # Create midpoint mapping from bin edges 
    midpoints[col] = { 
        "low": (edges[0] + edges[1]) / 2,
        "medium": (edges[1] + edges[2]) / 2,
        "high": (edges[2] + edges[3]) / 2,
    }

# Convert predicted categories back to numbers 
y_pred = [midpoints["diabetes_prev_agestd"][p] for p in predictions]
y_true = test_clean.loc[test_disc.dropna().index, "diabetes_prev_agestd"] 
metrics = evaluate_model(y_true, y_pred)
print(metrics)

# Run causal queries 
result = inference.query(
    variables=["diabetes_prev_agestd"],
    evidence={"health_exp_per_capita": "high"}
)
print(result)

# Save model 
with open("outputs/models/bn_model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("outputs/models/bn_bin_edges.pkl", "wb") as f:
    pickle.dump(bin_edges, f)