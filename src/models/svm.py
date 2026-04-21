import pandas as pd 
from src.models.evaluate import temporal_split 
from sklearn.preprocessing import StandardScaler 
from sklearn.svm import SVR
from src.models.evaluate import evaluate_model
import joblib
from sklearn.model_selection import GridSearchCV, GroupKFold

# Load feature matrix 
df = pd.read_parquet("data/model_ready/global_features.parquet") 

train, test = temporal_split(df) # Split data into training and testing data 

# Define features and target
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

# Scale features 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test) 

# Define parameter grid for the model
param_grid = { 
    'C':[100, 1000, 5000, 10000, 50000],
    'epsilon': [0.01, 0.05, 0.1, 0.2],
    'gamma': [0.0005, 0.001, 0.005, 0.01]
}

gkf = GroupKFold(n_splits=5)

# Wrap SVR inside GridSearchCV
grid = GridSearchCV( 
    SVR(kernel='rbf'),  # base model 
    param_grid,         # all combinations of parameters to try
    cv=gkf,             # how to split folds 
    scoring='neg_root_mean_squared_error',
    n_jobs=-1,           # use all cpu cores for parallel processing
    verbose=1
)

# Fit the model - pass groups so GroupKFold can split by country
grid.fit(X_train_scaled, y_train, groups=train["iso3_code"])

# Best parameters from grid search
model = grid.best_estimator_
print(f"Best params: {grid.best_params_}")                                                                                        
print(f"Support vectors: {model.n_support_.sum()} / {len(y_train)}") 

# Evaluate model on test data 
y_pred = model.predict(X_test_scaled)
metrics = evaluate_model(y_test, y_pred)
print(metrics)

# Save training medians (for null values in simulations)
joblib.dump(X_train.median(), "outputs/models/svm_training_medians.pkl")

# Save feature bounds for extrapolation clamping in simulations
feature_bounds = pd.DataFrame({'min': X_train.min(), 'max': X_train.max()})
joblib.dump(feature_bounds, "outputs/models/svm_feature_bounds.pkl") 

# Save model and scaler for future use
joblib.dump(model, "outputs/models/svm_model.pkl")
joblib.dump(scaler, "outputs/models/svm_scaler.pkl")



