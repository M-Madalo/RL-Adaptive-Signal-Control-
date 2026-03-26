import os
import subprocess
import pandas as pd

reward_variants = ["RF0", "RF1", "RF2", "RF3", "RF4"]
episodes = 50  # Quick training to generate complete metrics
main_training_script = "reinforcement_learningCAXX.py"

print("Generating complete training metrics for all variants...")
print("This will run a fresh training session (50 episodes) for each variant to ensure all metrics are collected from episode 1.")

for variant in reward_variants:
    print(f"\n=== Generating complete metrics for variant: {variant} ===")
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output', f'output_{variant}')
    os.makedirs(output_dir, exist_ok=True)

    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    # Run training with fresh start (no resume)
    print(f"Running {episodes} episodes for {variant} variant (fresh start)")
    subprocess.run([
        "python", main_training_script,
        "--episodes", str(episodes),
        "--reward_mode", variant
    ])

    print(f"Finished generating complete metrics for variant: {variant}")

print("\nAll complete metrics generated!")
print("Now you can run analyze_results.py to see the proper results with all metrics from episode 1.") 