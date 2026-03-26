import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# Load your summary data
summary_df = pd.read_csv("figures/final_results_table_detailed.csv")

# Configuration
metrics = ['travel_time', 'waiting_time', 'queue_length', 'stopped_vehicles', 'avg_speed']
metric_labels = {
    'travel_time': 'Travel Time',
    'waiting_time': 'Waiting Time',
    'queue_length': 'Queue Length',
    'stopped_vehicles': 'Stopped Vehicles',
    'avg_speed': 'Average Speed'
}
reward_variants = ['RF0', 'RF1', 'RF2', 'RF3', 'RF4']

# Plot preparation
bar_width = 0.15
x = np.arange(len(metrics))

# Define RGB colors for each variant
variant_colors = {
    'RF0': '#FF0000',  # Red
    'RF1': '#00FF00',  # Green
    'RF2': '#0000FF',  # Blue
    'RF3': '#FF8000',  # Orange
    'RF4': '#8000FF'   # Purple
}

# Wider figure to accommodate left-side legend
fig, ax = plt.subplots(figsize=(14, 6))

# Plot each variant's CVs
for i, variant in enumerate(reward_variants):
    cvs = []
    for metric in metrics:
        cv_col = f"{metric}_CV"
        std_col = f"{metric}_final_std"
        mean_col = f"{metric}_final_mean"

        if cv_col in summary_df.columns:
            value = summary_df.loc[summary_df['Variant'] == variant, cv_col].values
        elif std_col in summary_df.columns and mean_col in summary_df.columns:
            std = summary_df.loc[summary_df['Variant'] == variant, std_col].values
            mean = summary_df.loc[summary_df['Variant'] == variant, mean_col].values
            value = std / mean
        else:
            value = [np.nan]
        cvs.append(value[0] if len(value) > 0 else np.nan)

    ax.bar(x + i * bar_width, cvs, width=bar_width, label=variant, color=variant_colors[variant])

# Add instability threshold line
ax.axhline(0.2, color='red', linestyle='--', linewidth=1.2, label='Instability Threshold (CV=0.2)')

# X-axis settings
ax.set_xticks(x + bar_width * 2)
ax.set_xticklabels([metric_labels[m] for m in metrics], rotation=30)

# Labels and title
ax.set_ylabel("Coefficient of Variation (CV)")
ax.set_title("Stability of Reward Variants Across Performance Metrics")
ax.grid(True, axis='y', linestyle='--', alpha=0.5)

# Legend on the left, fully visible
ax.legend(loc='center left', bbox_to_anchor=(-0.45, 0.5), frameon=True)

# Adjust space to accommodate legend
fig.subplots_adjust(left=0.35)

# Save the updated figure
output_path = "figures/visual_stability_cv_bars_corrected.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"[Saved] Updated CV bar plot with clearly visible left-side legend: {output_path}")
