import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error 
import os 
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
import pandas as pd 
import shap 
import numpy as np 

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
def feature_importance(result, feature_names):
    
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

# Shap plot 
def plot_shap_beeswarm(model, X_test_scaled, feature_names, n_background=100, n_explain=100): 
    os.makedirs("outputs/figures/", exist_ok=True)

    # Background: summarize training distribution with k-means 
    background = shap.kmeans(X_test_scaled, n_background) 

    # Explainer: wraps any model using black-box perturbations 
    explainer = shap.KernelExplainer(model.predict, background)

    # Explain a subsample of test rows 
    X_explain = X_test_scaled[ :n_explain]
    shap_values = explainer.shap_values(X_explain) 

    #Convert scaled array back to a DataFrame so SHAP gets feature names 
    X_explain_df = pd.DataFrame(X_explain, columns=feature_names)

    # Generate beeswarm plot 
    shap.summary_plot(shap_values, X_explain_df, show=False)

    plt.title("SVM SHAP Feature Impact (Beeswarm)") 
    plt.tight_layout()
    plt.savefig("outputs/figures/svm_shap_summary.png", dpi=150, bbox_inches='tight') 
    plt.close() 

# Residual anaysis (Scatter plot)
def plot_residual_analysis(y_test, y_pred):
    os.makedirs("outputs/figures/", exist_ok=True)

    residuals = y_test - y_pred 

    fig, axes = plt.subplots(1, 2, figsize=(14, 5)) 

    # (a) Residuals vs Predicted
    axes[0].scatter(y_pred, residuals, alpha=0.5, edgecolors='k', linewidths=0.3)
    axes[0].axhline(0, color='r', linestyle='--', linewidth=1.5)
    axes[0].set_xlabel("Predicted Prevalence (%)")
    axes[0].set_ylabel("Residual (Actual - Predicted)")
    axes[0].set_title("Residuals vs Predicted")

    # (b) Residual vs distribution
    sns.histplot(residuals, kde=True, ax=axes[1])
    axes[1].axvline(0, color='r', linestyle='--', linewidth=1.5)
    axes[1].set_xlabel("Residual")
    axes[1].set_title("Residual Distribution")

    plt.tight_layout()
    plt.savefig("outputs/figures/svm_residual_analysis.png", dpi=150)
    plt.close()

def plot_partial_dependencies(model, X_train_scaled, feature_names, importance_result): 
    os.makedirs("outputs/figures/", exist_ok=True)

    top3_indices = importance_result.importances_mean.argsort()[-3:][::-1].tolist()

    display = PartialDependenceDisplay.from_estimator(
        model, 
        X_train_scaled,
        features=top3_indices,
        feature_names=feature_names,
        kind="both",
        subsample=500,
        random_state=42, 
        n_jobs=-1
    )

    display.figure__.suptitle("Partial Dependence Plots (Top 3 Features), y=1.02")
    plt.tight_layout()
    plt.savefig("outputs/figures/svm_partial_dependence.png", dpi=150,
  bbox_inches='tight') 
    plt.close() 