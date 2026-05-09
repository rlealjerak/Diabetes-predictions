import pickle 
import os 
import numpy as np 
import pandas as pd
import networkx as nx 
import matplotlib.pyplot as plt
from sklearn.metrics import mutual_info_score
from src.models.belief_network import load_bn_data
import seaborn as sns

# Define method to plot a network graph with edge strengths 
def plot_bn_graph():
    with open("outputs/models/bn_model.pkl", "rb") as f:
        model = pickle.load(f)

    # Load split data
    train, _, columns_to_use, train_disc, _, bin_edges, _ = load_bn_data()
    
    # Extract edges from the BN model 
    edges = list(model.edges())

    # Compute mutual information for each edge
    mi_scores = {}
    for (parent, child) in edges:
        # cat.codes converts "low"/"medium"/"high" to integers (0/1/2)
        # NaN categories come back as -1, so we filter those out
        parent_codes = train_disc[parent].cat.codes 
        child_codes = train_disc[child].cat.codes 
        valid = (parent_codes >= 0) & (child_codes >= 0)
        mi_scores[(parent, child)] = mutual_info_score(parent_codes[valid], child_codes[valid]) 

    # Build a directed graph     
    G = nx.DiGraph()
    G.add_edges_from(edges) 

    # Scale MI values to edge widths in the range [1, 6] for visibility 
    mi_values = np.array(list(mi_scores.values())) 
    mi_min, mi_max = mi_values.min(), mi_values.max() 
    if mi_max > mi_min: 
        edge_widths = 1 + 5 * (mi_values - mi_min) / (mi_max - mi_min) 
    else: 
        edge_widths = np.ones(len(mi_values)) * 2.0 

    # COmpute node positions using spring layout 
    # seed=42 makes ir reproducible; k controls spacing between nodes 
    pos = nx.spring_layout(G, seed=42, k=2) 

    #Plot the graph 
    os.makedirs("outputs/figures/", exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 8))

    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=2500, node_color="lightblue", alpha=0.9)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=9, font_weight="bold")
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        width=edge_widths.tolist(),
        edge_color="steelblue",
        arrows=True,
        arrowsize=20,
        connectionstyle="arc3,rad=0.1"
    )

    # Label each edge with its MI score (mi score = measure that explains the dependence between two variables) 
    edge_labels = {e: f"MI={v:.3f}" for e, v in mi_scores.items()} 
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax, font_size=7)

    ax.set_title("Belief Network: Structure and Edge Mutual Information", fontsize=13)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig("outputs/figures/bn_network_graph.png", dpi=150, bbox_inches="tight")
    plt.close() 

# Conditional Probability Heat Maps 
def plot_cpd_heatmap():
    with open("outputs/models/bn_model.pkl", "rb") as f:
        model = pickle.load(f) 

    cpd = model.get_cpds("diabetes_prev_agestd")
    parents = cpd.variables[1:] # parent order as stored in the CPD

    # Reshape flat CPT (3, 3, 3, 3, 3, 3) - dim 0 is diabetes, rest are parents
    values = cpd.values.reshape([3] * len(cpd.variables)) 

    # Fix the 3 less-important parents at "medium" (index 1); leave BMI and glucose free 
    free_parents = ["mean_bmi", "raised_blood_glucose_pct"] 
    idx = [slice(None)] # diabetes state dimension
    for p in parents: 
        idx.append(slice(None) if p in free_parents else 1) 

    sliced = values[tuple(idx)] # shape (3, 3, 3) - diabetes x BMI x glucose
    
    #figure out which free parent lands in which axis of sliced 
    free_in_order = [p for p in parents if p in free_parents]
    bmi_axis = 1 + free_in_order.index("mean_bmi")

    states= ["low", "medium", "high"] 
    os.makedirs("outputs/figures/", exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4)) 

    for i, d_state in enumerate(states):
        mat = sliced[i]   # shape (3, 3) 
        if bmi_axis == 2:    # ensure rows=BMI, cols=glucose for heatmap 
            mat = mat.T 

        df_h = pd.DataFrame(
            mat,
            index=["low", "medium", "high"],
            columns=["low", "medium", "high"],
        )

        sns.heatmap(df_h, ax=axes[i], annot=True, fmt=".3f",
                    cmap="YlOrRd", vmin=0, vmax=1, cbar=True)
        axes[i].set_title(f"P(diabetes = {d_state})")
        axes[i].set_xlabel("Blood Glucose Level")
        axes[i].set_ylabel("Mean BMI" if i == 0 else "")

    fig.suptitle(
        "CPD of Diabetes Prevalence Given BMI and Blood Glucose\n"
        "(Physical Inactivity, NCD Mortality, Health Expenditure fixed at 'medium')",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig("outputs/figures/bn_cpd_diabetes.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: outputs/figures/bn_cpd_diabetes.png")


if __name__ == "__main__":
    plot_bn_graph()
    plot_cpd_heatmap()
