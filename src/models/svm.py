import pandas as pd 
from src.models.evaluate import temporal_split 
from sklearn.preprocessing import StandardScaler 
from sklearn.svm import SVR
from src.models.evaluate import evaluate_model
import joblib

# Load feature matrix 
df = pd.read_parquet("data/model_ready/global_features.parquet") 

train, test = temporal_split(df) # Split data into training and testing data 

# Define features and target
feature_cols = [ c for c in df.columns if c not in ("iso3_code", "year",  "diabetes_prev_agestd")]
X_train = train[feature_cols]
X_test = test[feature_cols]
y_train = train["diabetes_prev_agestd"]
t_test = test["diabetes_prev_agestd"]

# Handle missing values by filling with median
X_train = X_train.fillna(X_train.median())
X_test = X_test.fillna(X_train.median()) # Use training median to avoid data leakage 

# Scale features 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test) 

# Train model 
model = SVR(kernel="rbf")
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
metrics = evaluate_model(t_test, y_pred)
print(metrics)

# Save model and scaler for future use
joblib.dump(model, "outputs/models/svm_model.pkl")
joblib.dump(scaler, "outputs/models/svm_scaler.pkl")



