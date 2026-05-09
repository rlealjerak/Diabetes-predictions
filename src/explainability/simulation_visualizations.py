import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path

SIM_CSV = Path("outputs/reports/simulation_results.csv")
FEATURE_PARQUET = Path("data/model_ready/global_features.parquet")
OUT_DIR = Path("outputs/figures")

SCENARIO_LABELS = {
    "baseline": "Baseline",
    "reduce_bmi_5pct": "Reduce BMI 5%",
    "reduce_glucose_10pct": "Reduce Glucose 10%",
    "increase_physical_activity": "Increase Activity",
    "combined_intervention": "Combined",
    "worst_case": "Worst Case",
    "increase_health_expenditure_per_capita": "↑ Health Exp",
}

SCENARIO_COLORS = {
    "baseline": "#4878CF",
    "reduce_bmi_5pct": "#6ACC65",
    "reduce_glucose_10pct": "#D65F5F",
    "increase_physical_activity": "#B47CC7",
    "combined_intervention": "#C4AD66",
    "worst_case": "#77BEDB",
    "increase_health_expenditure_per_capita": "#F7A35C",
}


def load_data():
    sim = pd.read_csv(SIM_CSV)
    hist = pd.read_parquet(FEATURE_PARQUET)[["iso3_code", "year", "diabetes_prev_agestd"]].dropna()
    return sim, hist


def plot_scenario_comparison(sim: pd.DataFrame):
    """Plot 4.1: Grouped bar chart of all scenarios per country at 2050."""
    df2050 = sim[sim["year"] == 2050].copy()
    countries = df2050["iso3_code"].unique()
    scenarios = list(SCENARIO_LABELS.keys())

    fig, axes = plt.subplots(2, 5, figsize=(20, 8), sharey=False)
    axes = axes.flatten()
    x = np.arange(len(scenarios))

    for i, country in enumerate(countries):
        ax = axes[i]
        country_data = df2050[df2050["iso3_code"] == country]
        prevalences = [
            country_data.loc[country_data["scenario"] == sc, "predicted_prevalence"].values[0]
            if sc in country_data["scenario"].values else np.nan
            for sc in scenarios
        ]
        colors = [SCENARIO_COLORS[sc] for sc in scenarios]
        ax.bar(x, prevalences, 0.6, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_title(country, fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([SCENARIO_LABELS[sc] for sc in scenarios], rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Prevalence (%)" if i % 5 == 0 else "")
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
        ax.grid(axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Diabetes Prevalence by Scenario at 2050", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    out = OUT_DIR / "sim_scenario_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_timelines(sim: pd.DataFrame, hist: pd.DataFrame):
    """Plot 4.2: Timeline per country — historical + projected scenarios."""
    countries = sim["iso3_code"].unique()

    for country in countries:
        fig, ax = plt.subplots(figsize=(10, 6))

        h = hist[hist["iso3_code"] == country].sort_values("year")
        last_hist_year = None
        if not h.empty:
            ax.plot(h["year"], h["diabetes_prev_agestd"], "ko-", linewidth=2,
                    markersize=4, label="Historical", zorder=5)
            last_hist_year = h["year"].max()
            ax.axvline(last_hist_year, color="gray", linestyle="--", alpha=0.6, linewidth=1)

        c_sim = sim[sim["iso3_code"] == country].sort_values("year")
        for scenario, label in SCENARIO_LABELS.items():
            sc_data = c_sim[c_sim["scenario"] == scenario]
            if sc_data.empty:
                continue
            if last_hist_year is not None:
                xs = [last_hist_year] + sc_data["year"].tolist()
                ys = [h["diabetes_prev_agestd"].iloc[-1]] + sc_data["predicted_prevalence"].tolist()
            else:
                xs = sc_data["year"].tolist()
                ys = sc_data["predicted_prevalence"].tolist()
            ls = "--" if scenario == "worst_case" else "-"
            ax.plot(xs, ys, marker="o", linestyle=ls, linewidth=1.8,
                    color=SCENARIO_COLORS[scenario], label=label, markersize=5)

        ax.set_title(f"{country} — Diabetes Prevalence Projections", fontsize=13, fontweight="bold")
        ax.set_xlabel("Year")
        ax.set_ylabel("Age-Standardised Prevalence (%)")
        ax.legend(loc="upper left", fontsize=8, framealpha=0.7)
        ax.grid(alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        out = OUT_DIR / f"sim_timeline_{country}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out}")


def plot_cross_country_heatmap(sim: pd.DataFrame):
    """Plot 4.3: Side-by-side heatmap — baseline vs combined_intervention."""
    scenarios_to_plot = ["baseline", "combined_intervention"]
    vmin = sim[sim["scenario"].isin(scenarios_to_plot)]["predicted_prevalence"].min()
    vmax = sim[sim["scenario"].isin(scenarios_to_plot)]["predicted_prevalence"].max()

    fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharey=True)
    for ax, scenario in zip(axes, scenarios_to_plot):
        pivot = sim[sim["scenario"] == scenario].pivot(
            index="iso3_code", columns="year", values="predicted_prevalence"
        )
        sns.heatmap(pivot, ax=ax, annot=True, fmt=".1f", cmap="YlOrRd",
                    linewidths=0.5, cbar_kws={"label": "Prevalence (%)"},
                    vmin=vmin, vmax=vmax)
        ax.set_title(SCENARIO_LABELS[scenario], fontsize=12, fontweight="bold")
        ax.set_xlabel("Year")
        ax.set_ylabel("Country" if ax is axes[0] else "")

    fig.suptitle("Diabetes Prevalence: Baseline vs Combined Intervention",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = OUT_DIR / "sim_cross_country_heatmap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_intervention_deltas(sim: pd.DataFrame):
    """Plot 4.4: Diverging horizontal bars — (scenario - baseline) at 2050."""
    df2050 = sim[sim["year"] == 2050].copy()
    baseline = df2050[df2050["scenario"] == "baseline"].set_index("iso3_code")["predicted_prevalence"]
    intervention_scenarios = [sc for sc in SCENARIO_LABELS if sc != "baseline"]
    countries = baseline.index.tolist()

    fig, axes = plt.subplots(1, len(intervention_scenarios), figsize=(24, 6), sharey=True)
    for ax, scenario in zip(axes, intervention_scenarios):
        sc_data = df2050[df2050["scenario"] == scenario].set_index("iso3_code")["predicted_prevalence"]
        deltas = sc_data.reindex(countries) - baseline.reindex(countries)
        colors = ["#2ECC71" if d <= 0 else "#E74C3C" for d in deltas]
        ax.barh(countries, deltas, color=colors, edgecolor="white")
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(SCENARIO_LABELS[scenario], fontsize=9, fontweight="bold")
        ax.set_xlabel("Δ Prevalence (%)")
        ax.grid(axis="x", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Country")
    fig.suptitle("Intervention Impact at 2050 vs Baseline  (Green = Reduction, Red = Increase)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = OUT_DIR / "sim_intervention_deltas.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    sim, hist = load_data()
    plot_scenario_comparison(sim)
    plot_timelines(sim, hist)
    plot_cross_country_heatmap(sim)
    plot_intervention_deltas(sim)
    print("All simulation visualizations complete.")
