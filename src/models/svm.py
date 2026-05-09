import pandas as pd 
from src.models.evaluate import temporal_split 
from sklearn.preprocessing import StandardScaler 
from sklearn.svm import SVR
from src.models.evaluate import evaluate_model
import joblib
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV, GroupKFold  
from sklearn.inspection import permutation_importance


# Load feature matrix 
# Define features and target 
# Load feature matrix 
# Do all those things inside a function for easier import 
def load_svm_data():
    df = pd.read_parquet("data/model_ready/global_features.parquet") 
    train, test = temporal_split(df) # Split data into training and testing data 

    selected_features = [ 
        "mean_bmi",
        "raised_blood_glucose_pct",
        "physical_inactivity_pct",
        "gdp_per_capita_ppp",
        "health_exp_per_capita",
        "life_expectancy",
        "pop_65plus_pct",
        "urban_pop_pct",
        "overweight_pct",
        "bmi_5yr_change",
        "physical_inactivity_5yr_change",
        "urban_pop_change_10yr",
        "health_exp_relative",
    ]
    
    X_train = train[selected_features]
    X_test = test[selected_features]
    y_train = train["diabetes_prev_agestd"]
    y_test = test["diabetes_prev_agestd"]

    # Handle missing values by filling with median
    X_train = X_train.fillna(X_train.median())
    X_test = X_test.fillna(X_train.median()) # Use training median to avoid data leakage 

    return X_train, X_test, y_train, y_test, selected_features, train 

def train_svm():
    X_train, X_test, y_train, y_test, feature_names, train = load_svm_data()

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    param_grid = {
        'C': [100, 1000, 5000, 10000, 50000],
        'epsilon': [0.01, 0.05, 0.1, 0.2],
        'gamma': [0.0005, 0.001, 0.005, 0.01]
    }

    gkf = GroupKFold(n_splits=5)

    grid = HalvingGridSearchCV(
        SVR(kernel='rbf'),
        param_grid,
        cv=gkf,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )

    grid.fit(X_train_scaled, y_train, groups=train["iso3_code"])

    model = grid.best_estimator_
    print(f"Best params: {grid.best_params_}")
    print(f"Support vectors: {model.n_support_.sum()} / {len(y_train)}")

    y_pred = model.predict(X_test_scaled)
    metrics = evaluate_model(y_test, y_pred)
    print(metrics)

    perm_result = permutation_importance(
        model, X_test_scaled, y_test,
        n_repeats=10, random_state=42, n_jobs=-1,
        scoring='neg_root_mean_squared_error'
    )

    training_medians = X_train.median()
    feature_bounds = pd.DataFrame({'min': X_train.min(), 'max': X_train.max()})

    joblib.dump(perm_result,      "outputs/models/svm_permutation_importance.pkl")
    joblib.dump(training_medians, "outputs/models/svm_training_medians.pkl")
    joblib.dump(feature_bounds,   "outputs/models/svm_feature_bounds.pkl")
    joblib.dump(model,            "outputs/models/svm_model.pkl")
    joblib.dump(scaler,           "outputs/models/svm_scaler.pkl")
    joblib.dump(metrics,          "outputs/models/svm_metrics.pkl")

    return {
        "model":            model,
        "scaler":           scaler,
        "X_train":          X_train,
        "X_test":           X_test,
        "X_train_scaled":   X_train_scaled,
        "X_test_scaled":    X_test_scaled,
        "y_train":          y_train,
        "y_test":           y_test,
        "feature_names":    feature_names,
        "perm_result":      perm_result,
        "metrics":          metrics,
        "training_medians": training_medians,
        "feature_bounds":   feature_bounds,
    }

if __name__ == "__main__":
    train_svm()



