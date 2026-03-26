import os
import torch
import pandas as pd
import glob

reward_variants = ["RF0", "RF1", "RF2", "RF3", "RF4"]

print("Extracting training metrics from existing checkpoints...")
print("This will create proper CSV files with all metrics from your previous training runs.")

for variant in reward_variants:
    print(f"\n=== Extracting metrics for variant: {variant} ===")
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output', f'output_{variant}')
    
    if not os.path.exists(output_dir):
        print(f"Output directory not found for {variant}, skipping...")
        continue
    
    # Find the latest checkpoint
    checkpoint_pattern = os.path.join(output_dir, "checkpoint_episode_*.pth")
    checkpoints = glob.glob(checkpoint_pattern)
    
    if not checkpoints:
        print(f"No checkpoints found for {variant}, skipping...")
        continue
    
    # Sort checkpoints by episode number and get the latest
    def extract_episode_number(filename):
        import re
        match = re.search(r'checkpoint_episode_(\d+)\.pth', filename)
        return int(match.group(1)) if match else 0
    
    latest_checkpoint = max(checkpoints, key=extract_episode_number)
    latest_episode = extract_episode_number(latest_checkpoint)
    
    print(f"Found latest checkpoint: {latest_checkpoint}")
    print(f"Extracting metrics from episode {latest_episode}")
    
    try:
        # Load the checkpoint with weights_only=False to allow loading numpy arrays and other data
        checkpoint = torch.load(latest_checkpoint, map_location='cpu', weights_only=False)
        
        # Extract metrics from checkpoint
        rewards_history = checkpoint.get('rewards_history', [])
        waiting_times_history = checkpoint.get('waiting_times_history', [])
        avg_speeds_history = checkpoint.get('avg_speeds_history', [])
        queue_lengths_history = checkpoint.get('queue_lengths_history', [])
        travel_times_history = checkpoint.get('travel_times_history', [])
        
        # Create dummy stopped_vehicles_history and epsilon_history if not present
        stopped_vehicles_history = checkpoint.get('stopped_vehicles_history', [0] * len(rewards_history))
        epsilon_history = checkpoint.get('epsilon_history', [0.1] * len(rewards_history))
        
        print(f"Extracted {len(rewards_history)} episodes of training data")
        
        # Create DataFrame with all metrics
        df = pd.DataFrame({
            'reward': rewards_history,
            'waiting_time': waiting_times_history,
            'avg_speed': avg_speeds_history,
            'queue_length': queue_lengths_history,
            'travel_time': travel_times_history,
            'stopped_vehicles': stopped_vehicles_history,
            'epsilon': epsilon_history
        })
        
        # Save to CSV
        csv_path = os.path.join(output_dir, 'training_metrics.csv')
        df.to_csv(csv_path, index=False)
        
        print(f"✓ Saved comprehensive metrics to {csv_path}")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Rows: {len(df)}")
        
        # Show some statistics
        if len(df) > 0:
            print(f"  Average reward: {df['reward'].mean():.2f}")
            print(f"  Average waiting time: {df['waiting_time'].mean():.2f}s")
            print(f"  Average speed: {df['avg_speed'].mean():.2f} mph")
            print(f"  Average queue length: {df['queue_length'].mean():.2f}")
            print(f"  Average travel time: {df['travel_time'].mean():.2f}s")
        
    except Exception as e:
        print(f"Error extracting metrics from {variant}: {e}")
        continue

print("\nAll metrics extraction finished!")
print("You can now run analyze_results.py to see the proper results.") 