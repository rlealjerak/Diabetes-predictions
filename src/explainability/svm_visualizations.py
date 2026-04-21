import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error 
import os 
from sklearn.inspection import permutation_importance 
import pandas as pd 

# Define function to create scatter plot 
def plot_actual_vs_predicted(y_test, y_pred): 
    os.makedirs("outputs/figures/", exist_ok=True)

    r2 = r2_score(y_test, y_pred) 
    rmse = mean_squared_error(y_test, y_pred) ** 0.5

    fig, ax = plt.subplots(figsize=(8,6)) 

    ax.scatter(y_test, y_pred, alpha=0.5, edgecolors='k', linewidths=0.3)

    ref_min = min(y_test.min(), y_pred.min()) 

    ref_max = max(y_test.max(), y_pred.max())
    ax.plot([ref_min, ref_max], [ref_min, ref_max], 'r--', linewidth=1.5) 

    ax.text(0.05, 0.92, f"R2 = {r2: .3f}\nRMSE = {rmse: .3f}", 
            transform=ax.transAxes, fontsize=11,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)) 
    
    ax.set_xlabel("Actual Prevalence (%)") 
    ax.set_ylabel("Predicted Prevalence (%)") 
    ax.set_title("SVM: Actual vs Predicted Diabetes Prevalence (Test Set)")

    plt.tight_layout()
    plt.savefig("outputs/figures/svm_actual_vs_predicted.png", dpi=150)
    plt.close()     

# Plot feature importance 
def feature_importance(model, X_test_scaled, y_test, feature_names):
    result = permutation_importance(model, X_test_scaled, y_test, n_repeats=30, random_state=42)
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': result.importances_mean,
        'std': result.importances_std
    }).sort_values('importance', ascending=False).head(15)

    importance_df = importance_df.sort_values('importance', ascending=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(importance_df['feature'], importance_df['importance'],
            xerr=importance_df['std'], align='center', alpha=0.7) 
    
    ax.set_xlabel("Mean Decrease in Performance (RMSE)")
    ax.set_ylabel("Feature")
    ax.set_title("SVM Feature Importance (Permutation, Test Set)") 

    plt.tight_layout() 
    plt.savefig("outputs/figures/svm_feature_importance.png", dpi=150)
    plt.close()