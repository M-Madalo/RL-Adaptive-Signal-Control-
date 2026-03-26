import os
import itertools
from scipy.stats import ttest_ind, friedmanchisquare
import pandas as pd

# === Load per-episode metric data for each variant ===
base_path = "output"
reward_variants = ["RF0", "RF1", "RF2", "RF3", "RF4"]
variant_dfs = {}

for variant in reward_variants:
    file_path = os.path.join(base_path, f"output_{variant}", "training_metrics.csv")
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        variant_dfs[variant] = df
    else:
        print(f"[Warning] File missing: {file_path}")

# === Define readable metric labels ===
metrics_labels = {
    'waiting_time': 'Average Waiting Time (s)',
    'queue_length': 'Average Queue Length (vehicles)',
    'avg_speed': 'Average Speed (mph)',
    'travel_time': 'Average Travel Time (s)',
    'stopped_vehicles': 'Stopped Vehicles (count)',
    'reward': 'Episode Reward'
}

# === A. Welch's t-test (pairwise comparisons) ===
ttest_results = []

for metric_key, metric_label in metrics_labels.items():
    pairs = list(itertools.combinations(reward_variants, 2))
    for v1, v2 in pairs:
        if v1 not in variant_dfs or v2 not in variant_dfs:
            continue
        df1 = variant_dfs[v1]
        df2 = variant_dfs[v2]

        if metric_key not in df1.columns or metric_key not in df2.columns:
            continue

        x1 = df1[metric_key].iloc[-20:]
        x2 = df2[metric_key].iloc[-20:]

        stat, p = ttest_ind(x1, x2, equal_var=False)
        ttest_results.append({
            "Metric": metric_label,
            "Variant A": v1,
            "Variant B": v2,
            "t-stat": round(stat, 3),
            "p-value": format(p, ".3e"),
            "Significant (p < 0.05)": "✓" if p < 0.05 else ""
        })

# Save t-test results
ttest_df = pd.DataFrame(ttest_results)
ttest_path = "figures/ttest_pairwise_results.csv"
ttest_df.to_csv(ttest_path, index=False)
print(f"[Saved] Welch's t-test results to: {ttest_path}")

# === B. Friedman test (all 5 variants) ===
friedman_results = []

for metric_key, metric_label in metrics_labels.items():
    final_data = []
    included = []

    for variant in reward_variants:
        if variant in variant_dfs and metric_key in variant_dfs[variant].columns:
            x = variant_dfs[variant][metric_key].iloc[-20:]
            if len(x) == 20:
                final_data.append(x.values)
                included.append(variant)

    if len(final_data) < 3:
        continue

    aligned = list(zip(*final_data))  # transpose for episodes × variants
    stat, p = friedmanchisquare(*aligned)

    friedman_results.append({
        "Metric": metric_label,
        "Included Variants": ", ".join(included),
        "Chi²": round(stat, 3),
        "p-value": format(p, ".3e"),
        "Significant (p < 0.05)": "✓" if p < 0.05 else ""
    })

# Save Friedman results
friedman_df = pd.DataFrame(friedman_results)
friedman_path = "figures/friedman_anova_results.csv"
friedman_df.to_csv(friedman_path, index=False)
print(f"[Saved] Friedman test results to: {friedman_path}")
