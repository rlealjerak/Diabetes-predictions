import os

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np 
from sklearn.model_selection import GroupKFold
import json 

# Metrics functions 
def evaluate_model(y_true, y_pred):
    return{ 
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "mape": np.mean(np.abs((y_true - y_pred) / y_true)) * 100  
    }

# Temporal split to train on older data and test on more recent data 
def temporal_split(df, holdout_years=[2014, 2015, 2016]):
    train = df[~df["year"].isin(holdout_years)]
    test = df[df["year"].isin(holdout_years)]
    return train, test

# Cross validation function 
def group_kfold_cv(model, X, y, groups, n_splits=5):
    gkf = GroupKFold(n_splits=n_splits)
    results = []
    for train_idx, test_idx in gkf.split(X, y, groups):
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = model.predict(X.iloc[test_idx])
        results.append(evaluate_model(y.iloc[test_idx], preds))
    # Average results across folds
    return {k: np.mean([r[k] for r in results]) for k in results[0]} 

# Save results to a json file 
def save_results(results, path="outputs/reports/model_comparison.json"):                                                                                 
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:                                                                                                                           
        json.dump(results, f, indent=2)