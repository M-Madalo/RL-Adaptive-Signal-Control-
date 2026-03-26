import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# === Settings ===
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
N_final_episodes = 20

# --- Prepare to collect results ---
results = {}
variant_dfs = {}

# --- Load and summarize data ---
for variant in reward_variants:
    csv_path = os.path.join(base_path, f"output_{variant}", f"{variant}training_metrics.csv")
    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found. Skipping {variant}.")
        continue

    df = pd.read_csv(csv_path)
    print(f"Loaded {variant}: {df.shape[0]} rows, columns: {list(df.columns)}")
    variant_dfs[variant] = df
    results[variant] = {}
    
    for metric in metrics_labels.keys():
        if metric not in df.columns:
            print(f"  [Metric Missing] '{metric}' not in {variant} data. Filling with NaN.")
            results[variant][metric] = {'mean_final': np.nan, 'std_final': np.nan, 'mean_full': np.nan, 'std_full': np.nan}
            continue

        # Calculate metrics for the final N episodes
        final_data = df[metric].iloc[-N_final_episodes:]
        results[variant][metric] = {
            'mean_final': final_data.mean(),
            'std_final': final_data.std(),
            'mean_full': df[metric].mean(),
            'std_full': df[metric].std()
        }

# --- 1. Final results table ---
print("\n=== FINAL RESULTS TABLE (Final 20 Episodes vs. Full Run) ===")
header = f"{'Variant':<12} | {'Metric':<15} | {'Final 20 Avg ± Std':<20} | {'Full Run Avg ± Std':<20}"
print(header)
print("-" * len(header))

for metric, label in metrics_labels.items():
    for variant in reward_variants:
        if variant in results and metric in results[variant]:
            res = results[variant][metric]
            
            final_str = f"{res.get('mean_final', 0):.2f} ± {res.get('std_final', 0):.2f}"
            full_str = f"{res.get('mean_full', 0):.2f} ± {res.get('std_full', 0):.2f}"

            # Handle potential NaN values from missing data
            if pd.isna(res.get('mean_final')): final_str = "nan ± nan"
            if pd.isna(res.get('mean_full')): full_str = "nan ± nan"

            print(f"{variant:<12} | {label.split('(')[0].strip():<15} | {final_str:<20} | {full_str:<20}")
    print("-" * len(header))

# --- Save Summary Table to CSV ---
summary_data = []
for variant in reward_variants:
    if variant not in results: continue
    row = {'Variant': variant}
    for metric in metrics_labels.keys():
        if metric in results[variant]:
            res = results[variant][metric]
            row[f'{metric}_final_mean'] = res.get('mean_final')
            row[f'{metric}_final_std'] = res.get('std_final')
            row[f'{metric}_full_mean'] = res.get('mean_full')
            row[f'{metric}_full_std'] = res.get('std_full')
    summary_data.append(row)

if summary_data:
    summary_df = pd.DataFrame(summary_data)
    summary_csv_path = os.path.join(fig_dir, "final_results_table_detailed.csv")
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"\n[Saved] Detailed summary table to {summary_csv_path}")

# --- TRB-style figure settings ---
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'lines.linewidth': 2,
    'axes.spines.top': False,
    'axes.spines.right': False
})

# Define RGB colors for each variant
variant_colors = {
    'RF0': '#FF0000',  # Red
    'RF1': '#00FF00',  # Green
    'RF2': '#0000FF',  # Blue
    'RF3': '#FF8000',  # Orange
    'RF4': '#8000FF'   # Purple
}

# --- 1. Combined plot per metric (all variants) ---
for metric, label in metrics_labels.items():
    plt.figure(figsize=(8, 5))
    has_data = False
    for variant in reward_variants:
        if variant not in variant_dfs:
            continue
        df = variant_dfs[variant]
        if metric not in df.columns:
            continue
        plt.plot(
            np.arange(1, len(df[metric]) + 1),
            df[metric],
            label=variant.capitalize(),
            color=variant_colors.get(variant, '#808080')  # Gray fallback
        )
        has_data = True
    
    if has_data:
        plt.title(label)
        plt.xlabel("Episode")
        plt.ylabel(label)
        plt.grid(True, linestyle='--', linewidth=0.4, alpha=0.7)
        plt.legend(fontsize='small', loc='upper right')
        plt.tight_layout()
        fname = os.path.join(fig_dir, f"compare_{metric}.png")
        plt.savefig(fname, dpi=300)
        plt.close()
        print(f"[Saved] {fname}")
    else:
        plt.close()
        print(f"[Skipped] {metric} - no data available")

# --- 2. Individual plots per variant (all metrics) ---
for variant in reward_variants:
    if variant not in variant_dfs:
        continue
    df = variant_dfs[variant]

    fig, ax = plt.subplots(figsize=(12, 6))
    has_data = False
    for metric, label in metrics_labels.items():
        if metric in df.columns:
            ax.plot(df[metric], label=label, linewidth=2)
            has_data = True
    
    if has_data:
        ax.set_title(f"Training Metrics – {variant.capitalize()}")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Metric Value")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
        
        # Place legend to the right of the plot
        lgd = ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize='small')
        
        fname = os.path.join(fig_dir, f"{variant}_metrics_overview.png")
        # Use bbox_inches='tight' and pass legend object to ensure it's saved correctly
        plt.savefig(fname, dpi=300, bbox_extra_artists=(lgd,), bbox_inches='tight')
        plt.close(fig)
        print(f"[Saved] {fname}")
    else:
        # If there's a figure with no data, close it
        if has_data is False:
            plt.close()
        print(f"[Skipped] {variant} - no data available")

# --- 3. Bar chart for final results ---
print("\n--- Generating Final Results Bar Charts ---")

for metric, label in metrics_labels.items():
    
    # Collect means and stds, skipping any variants that are missing or have NaN values
    variants_with_data = [
        v for v in reward_variants 
        if v in results and 'mean_final' in results[v][metric] and not np.isnan(results[v][metric]['mean_final'])
    ]
    if not variants_with_data:
        print(f"[Skipped] Bar chart for {metric} - no data available.")
        continue

    means = [results[v][metric]['mean_final'] for v in variants_with_data]
    stds = [results[v][metric]['std_final'] for v in variants_with_data]
    
    plt.figure(figsize=(10, 6))
    x_pos = np.arange(len(variants_with_data))
    
    # Create bars with error bars using RGB colors
    bars = plt.bar(x_pos, means, yerr=stds, align='center', alpha=0.85, ecolor='black', capsize=10, 
                   color=[variant_colors.get(v, '#808080') for v in variants_with_data])
    
    plt.ylabel(label)
    plt.xticks(x_pos, variants_with_data)
    plt.title(f'Final Performance Comparison: {label}')
    plt.grid(True, linestyle='--', axis='y', alpha=0.7)
    
    # Add the value label on top of each bar for clarity
    plt.bar_label(bars, fmt='%.2f', padding=3, color='black', fontsize=10)
    
    plt.tight_layout()
    fname = os.path.join(fig_dir, f"barchart_{metric}_comparison.png")
    plt.savefig(fname, dpi=300)
    plt.close()
    print(f"[Saved] {fname}")

# --- Export summary as CSV ---
summary_path = os.path.join(fig_dir, "final_results_table.csv")
summary_df = pd.DataFrame.from_dict({
    variant: {
        f"{metric}_mean": results[variant][metric]["mean_final"]
        for metric in metrics_labels
    } | {
        f"{metric}_std": results[variant][metric]["std_final"]
        for metric in metrics_labels
    }
    for variant in results
}, orient='index')
summary_df.to_csv(summary_path)
print(f"\n[Saved] Summary table to {summary_path}")

# --- Print recommendations ---
print("\n=== RECOMMENDATIONS ===")
missing_columns = set()
for variant in reward_variants:
    if variant in variant_dfs:
        df = variant_dfs[variant]
        for column in metrics_labels:
            if column not in df.columns:
                missing_columns.add(column)

if missing_columns:
    print(f"Missing columns detected: {missing_columns}")
    print("To fix this, run the training script again with proper metrics collection:")
    print("python regenerate_metrics.py")
    print("OR")
    print("python train_all_variants.py")
else:
    print("All metrics are available! Analysis complete.")
