import pandas as pd
from src.models.evaluate import temporal_split, evaluate_model
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
import pickle
import joblib

def discretize(train, test, columns):
    train_clean = train[columns].dropna()
    test_clean = test[columns].dropna()

    train_disc = train_clean.copy()
    test_disc = test_clean.copy()

    bin_edges = {}
    for col in columns:
        train_disc[col], bins = pd.qcut(train_clean[col], q=3, labels=["low", "medium", "high"], retbins=True)
        bin_edges[col] = bins.copy()
        cut_bins = bins.copy()
        cut_bins[0] = -float("inf")
        cut_bins[-1] = float("inf")
        test_disc[col] = pd.cut(test_clean[col], bins=cut_bins, labels=["low", "medium", "high"])

    return train_disc, test_disc, bin_edges

# Load data and discretize 
def load_bn_data(): 
    df = pd.read_parquet("data/model_ready/global_features.parquet")
    train, test = temporal_split(df)

    columns_to_use = ["mean_bmi", "raised_blood_glucose_pct", "physical_inactivity_pct", "ncd_mortality_prob", "diabetes_prev_agestd", "health_exp_per_capita", "gdp_per_capita_ppp", "urban_pop_pct"]

    train_disc, test_disc, bin_edges = discretize(train, test, columns_to_use)
    test_clean = test[columns_to_use].dropna()

    print(f"Train rows after discretize: {len(train_disc.dropna())}")
    print(f"Test rows after discretize: {len(test_disc.dropna())}")

    return train, test, columns_to_use, train_disc, test_disc, bin_edges, test_clean

def train_bn():
    train, test, columns_to_use, train_disc, test_disc, bin_edges, test_clean = load_bn_data()

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

    model.fit(train_disc.dropna(), estimator=MaximumLikelihoodEstimator)

    inference = VariableElimination(model)

    predictions = []
    for _, row in test_disc.dropna().iterrows():
        evidence = {col: row[col] for col in columns_to_use if col != "diabetes_prev_agestd"}
        pred = inference.map_query(variables=["diabetes_prev_agestd"], evidence=evidence)
        predictions.append(pred["diabetes_prev_agestd"])

    midpoints = {}
    for col, edges in bin_edges.items():
        midpoints[col] = {
            "low":    (edges[0] + edges[1]) / 2,
            "medium": (edges[1] + edges[2]) / 2,
            "high":   (edges[2] + edges[3]) / 2,
        }

    y_pred = [midpoints["diabetes_prev_agestd"][p] for p in predictions]
    y_true = test_clean.loc[test_disc.dropna().index, "diabetes_prev_agestd"]
    metrics = evaluate_model(y_true, y_pred)
    print(metrics)

    causal_queries = {}
    for feat in ["mean_bmi", "raised_blood_glucose_pct", "physical_inactivity_pct", "health_exp_per_capita"]:
        q_high = inference.query(["diabetes_prev_agestd"], evidence={feat: "high"})
        q_low  = inference.query(["diabetes_prev_agestd"], evidence={feat: "low"})
        states = q_high.state_names["diabetes_prev_agestd"]
        causal_queries[feat] = {
            "high": dict(zip(states, q_high.values)),
            "low":  dict(zip(states, q_low.values)),
        }

    with open("outputs/models/bn_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("outputs/models/bn_bin_edges.pkl", "wb") as f:
        pickle.dump(bin_edges, f)
    joblib.dump(causal_queries, "outputs/models/bn_causal_queries.pkl")
    joblib.dump(metrics,        "outputs/models/bn_metrics.pkl")

    return {
        "model":          model,
        "inference":      inference,
        "bin_edges":      bin_edges,
        "metrics":        metrics,
        "causal_queries": causal_queries,
    }

if __name__ == "__main__":
    train_bn()