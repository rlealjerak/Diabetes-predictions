import os 
import pickle
from datetime import datetime 
import joblib
import numpy as np 
import pandas as pd 

# ---- Configuration 
TOP_FEATURES = 5
REPORT_PATH  = "outputs/reports/pipeline_report.txt"

os.makedirs("outputs/models",  exist_ok=True)
os.makedirs("outputs/figures", exist_ok=True)
os.makedirs("outputs/reports", exist_ok=True)

# Step 1 of the pipeline -- Run SVM 
def run_svm(): 
    from src.models.svm import train_svm
    print(" [Step 1] Training SVM model...")
    result = train_svm()
    print(f" [Step 1] SVM training complete. Metrics: {result['metrics']}")
    return result 

# Step 2 of the pipeline -- Run BN
def run_bn():
    from src.models.belief_network import train_bn
    print("[Step 2] Training Belief Network model...") 
    result = train_bn()
    print(f" [Step 2] BN training complete. Metrics: {result['metrics']}")
    return result 

# Step 3 of the pipeline -- Run Simulations
def run_simulations():
    from src.simulations.engine import run_simulations as engine_run
    print("[Step 3] Running simulations...")
    result = engine_run()
    print(f"[Step 3] Simulations complete. {len(result)} rows saved.")
    return result

# Step 4 of the pipeline -- Generate visualizations 
def generate_visualizations(svm_result):
    from src.explainability.svm_visualizations import(
        plot_actual_vs_predicted,
        feature_importance,
        plot_residual_analysis,
        plot_partial_dependencies,
    )
    from src.explainability.bn_visualizations import plot_bn_graph, plot_cpd_heatmap 
    from src.explainability.simulation_visualizations import (
        load_data,
        plot_scenario_comparison,
        plot_timelines,
        plot_cross_country_heatmap,
        plot_intervention_deltas,
    )

    print( "[Step 4] Generating svm visualizations...") 
    y_pred = svm_result["model"].predict(svm_result["X_test_scaled"])
    plot_actual_vs_predicted(svm_result["y_test"], y_pred)
    feature_importance(svm_result["perm_result"], svm_result["feature_names"])
    plot_residual_analysis(svm_result["y_test"], y_pred)
    plot_partial_dependencies(
        svm_result["model"],
        svm_result["X_train_scaled"],
        svm_result["feature_names"],
        svm_result["perm_result"],
    )     

    print("[Step 4] Generating BN visualizations...")
    plot_bn_graph()
    plot_cpd_heatmap()

    print("[Step 4] Generating simulation visualizations...")
    sim, hist = load_data()
    plot_scenario_comparison(sim)
    plot_timelines(sim, hist)
    plot_cross_country_heatmap(sim)
    plot_intervention_deltas(sim) 

    print("[Step 4] All visualizations generated.")


# Step 5 of the pipeline -- Generate report
def generate_report(svm_result, bn_result, sim_result):
    print("[Step 5] Generating report...")

    lines = []
    lines.append("=" * 60)
    lines.append("PIPELINE REPORT")
    lines.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 60)

    lines.append("\n--- SVM Metrics ---")
    for k, v in svm_result["metrics"].items():
        lines.append(f"  {k.upper():<6}: {v:.4f}")

    lines.append("\n--- Top 10 Features by Permutation Importance ---")
    perm   = svm_result["perm_result"]
    names  = svm_result["feature_names"]
    ranked = sorted(zip(names, perm.importances_mean), key=lambda x: x[1], reverse=True)
    for i, (feat, imp) in enumerate(ranked[:10], 1):
        lines.append(f"  {i:2}. {feat:<35} {imp:.4f}")

    lines.append("\n--- Belief Network Metrics ---")
    for k, v in bn_result["metrics"].items():
        lines.append(f"  {k.upper():<6}: {v:.4f}")

    lines.append("\n--- Strongest BN Causal Relationships ---")
    for feat, q in bn_result["causal_queries"].items():
        delta = q["high"].get("high", 0) - q["low"].get("high", 0)
        lines.append(f"  {feat:<35} delta P(diabetes='high'): {delta:+.3f}")

    lines.append("\n--- Simulation Summary at 2050 ---")
    summary = sim_result[sim_result["year"] == 2050].pivot_table(
        index="iso3_code", columns="scenario", values="predicted_prevalence"
    )
    lines.append(summary.to_string())

    report = "\n".join(lines)
    with open(REPORT_PATH, "w") as f:
        f.write(report)

    print(report)
    print(f"\n[Step 5] Report saved to {REPORT_PATH}")


if __name__ == "__main__":
    print("=" * 60)
    print("STARTING FULL PIPELINE")
    print("=" * 60)

    svm_result = None
    bn_result  = None
    sim_result = None

    try:
        print("\n--- Step 1: Train SVM ---")
        svm_result = run_svm()
    except Exception as e:
        print(f"[FAILED] Step 1: {e}")

    try:
        print("\n--- Step 2: Train Belief Network ---")
        bn_result = run_bn()
    except Exception as e:
        print(f"[FAILED] Step 2: {e}")

    try:
        print("\n--- Step 3: Run Simulations ---")
        sim_result = run_simulations()
    except Exception as e:
        print(f"[FAILED] Step 3: {e}")

    try:
        print("\n--- Step 4: Generate Visualizations ---")
        if svm_result:
            generate_visualizations(svm_result)
        else:
            print("[SKIPPED] Step 4 requires Step 1 to succeed.")
    except Exception as e:
        print(f"[FAILED] Step 4: {e}")

    try:
        print("\n--- Step 5: Generate Report ---")
        if svm_result and bn_result and sim_result is not None:
            generate_report(svm_result, bn_result, sim_result)
        else:
            print("[SKIPPED] Step 5 requires Steps 1, 2, and 3 to succeed.")
    except Exception as e:
        print(f"[FAILED] Step 5: {e}")

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)


