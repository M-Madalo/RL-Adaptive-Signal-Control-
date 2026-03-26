import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# === Setup ===
base_path = "output"
fig_dir = "figures"
os.makedirs(fig_dir, exist_ok=True)

reward_variants = ["RF0", "RF1", "RF2", "RF3", "RF4"]
metrics_labels = {
    'waiting_time': 'Average Waiting Time (s)',
    'queue_length': 'Average Queue Length (vehicles)',
    'avg_speed': 'Average Speed (mph)',
    'travel_time': 'Average Travel Time (s)',
    'stopped_vehicles': 'Stopped Vehicles (count)',
    'reward': 'Episode Reward'
}

# === Load Data ===
variant_dfs = {}
for variant in reward_variants:
    csv_path = os.path.join(base_path, f"output_{variant}", f"{variant}training_metrics.csv")
    if not os.path.exists(csv_path):
        print(f"[Missing] {csv_path}")
        continue
    df = pd.read_csv(csv_path)
    variant_dfs[variant] = df

# === Plot Colors ===
metric_colors = {
    'waiting_time': '#FF0000',      # Red
    'queue_length': '#00FF00',      # Green
    'avg_speed': '#0000FF',         # Blue
    'travel_time': '#FF8000',       # Orange
    'stopped_vehicles': '#8000FF',  # Purple
    'reward': '#FF0080'             # Magenta
}

# === Plot Config ===
ncols = 3
nrows = (len(reward_variants) + ncols - 1) // ncols
rolling_window = 10

# === 1. Medium-Large Smoothed Plot (No Title) ===
print("\n--- Generating Medium-Large Smoothed Plot Without Title ---")

fig_width = 13
fig_height = 9
fig = plt.figure(figsize=(fig_width, fig_height))
gs = plt.GridSpec(nrows, ncols, hspace=0.6, wspace=0.45, left=0.25)

legend_handles = []

for idx, variant in enumerate(reward_variants):
    if variant not in variant_dfs:
        continue
    df = variant_dfs[variant]
    ax = fig.add_subplot(gs[idx])
    for metric, label in metrics_labels.items():
        if metric in df.columns:
            smoothed = df[metric].rolling(window=rolling_window, min_periods=1).mean()
            line, = ax.plot(smoothed, label=label, color=metric_colors.get(metric, '#808080'), linewidth=2)
            if idx == 0:
                legend_handles.append(line)
    ax.set_title(f"{variant}", fontsize=13)
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Metric Value", fontsize=12)
    ax.tick_params(axis='both', labelsize=11)
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)

legend_ax = fig.add_axes([0.01, 0.12, 0.2, 0.8])
legend_ax.axis('off')
legend_ax.legend(handles=legend_handles, loc='center left', fontsize='medium', frameon=False)

path1 = os.path.join(fig_dir, "summary_grid_smoothed_training_metrics_legend_left_larger_notitle.png")
plt.savefig(path1, dpi=400, bbox_inches='tight')
plt.close()
print(f"[Saved] {path1}")

# === 2. Large-Scale Publication Plot (No Title) ===
print("\n--- Generating Large Publication Plot Without Title ---")

fig_width = 17
fig_height = 12
fig = plt.figure(figsize=(fig_width, fig_height))
gs = plt.GridSpec(nrows, ncols, hspace=0.6, wspace=0.45, left=0.25)

legend_handles = []

for idx, variant in enumerate(reward_variants):
    if variant not in variant_dfs:
        continue
    df = variant_dfs[variant]
    ax = fig.add_subplot(gs[idx])
    for metric, label in metrics_labels.items():
        if metric in df.columns:
            smoothed = df[metric].rolling(window=rolling_window, min_periods=1).mean()
            line, = ax.plot(smoothed, label=label, color=metric_colors.get(metric, '#808080'), linewidth=2)
            if idx == 0:
                legend_handles.append(line)
    ax.set_title(f"{variant}", fontsize=15, weight='bold')
    ax.set_xlabel("Episode", fontsize=13)
    ax.set_ylabel("Metric Value", fontsize=13)
    ax.tick_params(axis='both', labelsize=12)
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)

legend_ax = fig.add_axes([0.01, 0.12, 0.2, 0.8])
legend_ax.axis('off')
legend_ax.legend(
    handles=legend_handles,
    loc='center left',
    fontsize=12,
    frameon=True,
    framealpha=0.9,
    edgecolor='gray'
)

path2 = os.path.join(fig_dir, "summary_grid_smoothed_training_metrics_large_left_larger_notitle.png")
plt.savefig(path2, dpi=500, bbox_inches='tight')
plt.close()
print(f"[Saved] {path2}")
