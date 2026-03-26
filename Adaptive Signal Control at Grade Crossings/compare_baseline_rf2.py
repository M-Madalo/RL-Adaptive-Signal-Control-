import pandas as pd
import numpy as np

# Load data
rf2 = pd.read_csv('output/output_RF2/RF2training_metrics.csv')
base = pd.read_csv('baseline_fixed_traffic_lights.csv')

# Metrics to compare
cols = ['avg_waiting_time','avg_speed','avg_queue_length','avg_stopped_vehicles','avg_travel_time']

# Get last 50 and 20 episodes
rf2_50 = rf2.tail(50)
rf2_20 = rf2.tail(20)

# Compute means
rf2_50_means = rf2_50[['waiting_time','avg_speed','queue_length','stopped_vehicles','travel_time']].mean()
rf2_20_means = rf2_20[['waiting_time','avg_speed','queue_length','stopped_vehicles','travel_time']].mean()
base_means = base[cols].mean()

# Compute improvement (%)
improvement_50 = 100 * (base_means.values - rf2_50_means.values) / base_means.values
improvement_20 = 100 * (base_means.values - rf2_20_means.values) / base_means.values

# Prepare output DataFrame
out = pd.DataFrame({
    'Metric': cols,
    'Baseline_Mean': base_means.values,
    'RL_Mean_50episodes': rf2_50_means.values,
    'RL_Mean_20episodes': rf2_20_means.values,
    'Improvement_50episodes_%': improvement_50,
    'Improvement_20episodes_%': improvement_20
})

# Save to CSV
out.to_csv('baseline_vs_rf2_comparison.csv', index=False)

# Display the result
print("Comparison Results:")
print("=" * 80)
print(out.to_string(index=False))
print("\n" + "=" * 80)
print(f"Baseline episodes: {len(base)}")
print(f"RF2 last 50 episodes: {len(rf2_50)}")
print(f"RF2 last 20 episodes: {len(rf2_20)}")
print("=" * 80) 