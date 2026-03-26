import pandas as pd
import os

# Define the mapping of reward variant filenames with correct paths
variant_files = {
    'RF0': 'output/output_RF0/RF0training_metrics.csv',
    'RF1': 'output/output_RF1/RF1training_metrics.csv',
    'RF2': 'output/output_RF2/RF2training_metrics.csv',
    'RF3': 'output/output_RF3/RF3training_metrics.csv',
    'RF4': 'output/output_RF4/RF4training_metrics.csv'
}

# Define the metrics to analyze
metrics = ['waiting_time', 'queue_length', 'avg_speed', 'travel_time', 'stopped_vehicles']

# Store summary statistics
summary_data = []

# Load and process each variant file
for variant, file in variant_files.items():
    if not os.path.exists(file):
        print(f"⚠️  Warning: {file} not found, skipping {variant}")
        continue
        
    try:
        df = pd.read_csv(file)
        print(f"✅ Loaded {variant}: {len(df)} episodes")

        # Ensure only the last 20 episodes are used
        last20 = df.tail(20)

        for metric in metrics:
            if metric not in last20.columns:
                print(f"⚠️  Warning: {metric} not found in {variant}")
                continue

            mean_val = last20[metric].mean()
            std_val = last20[metric].std()

            summary_data.append({
                'Variant': variant,
                'Metric': metric,
                'Mean': mean_val,
                'Std': std_val
            })
    except Exception as e:
        print(f"❌ Error processing {variant}: {e}")
        continue

# Create summary DataFrame
if not summary_data:
    print("❌ No data found! Check if the training metrics files exist.")
    exit(1)

summary_df = pd.DataFrame(summary_data)

# === Rank and extract Top 3 Variants per Metric ===
top3_records = []

for metric in metrics:
    metric_data = summary_df[summary_df['Metric'] == metric]
    if len(metric_data) == 0:
        continue
        
    ascending = metric not in ['avg_speed']  # lower is better for others
    ranked = metric_data.sort_values(by='Mean', ascending=ascending).reset_index(drop=True)

    for rank, row in ranked.head(3).iterrows():
        top3_records.append({
            'Metric': metric.replace('_', ' ').title(),
            'Rank': rank + 1,
            'Variant': row['Variant'],
            'Mean': round(row['Mean'], 2),
            'Std': round(row['Std'], 2)
        })

# Convert to DataFrame
top3_df = pd.DataFrame(top3_records)

# Save to CSV
top3_df.to_csv("top3_reward_variants_per_metric.csv", index=False)
print(f"✅ Saved top 3 rankings to: 'top3_reward_variants_per_metric.csv'")

print(f"\n📊 Top 3 Variants per Metric:")
print(top3_df)
