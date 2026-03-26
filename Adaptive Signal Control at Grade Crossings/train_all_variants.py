import os
import subprocess
import pandas as pd

reward_variants = ["RF0", "RF1", "RF2", "RF3", "RF4"]
episodes = 300
main_training_script = "reinforcement_learningCAXX.py"

for variant in reward_variants:
    print(f"\n=== Training for variant: {variant} ===")
    print(f"Running {episodes} episodes for {variant} variant")
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output', f'output_{variant}')
    os.makedirs(output_dir, exist_ok=True)

    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    subprocess.run([
        "python", main_training_script,
        "--episodes", str(episodes),
        "--reward_mode", variant
    ])

    print(f"Finished training for variant: {variant}\n")

print("\nAll variants finished!")
print("Each variant has been trained for 300 episodes.")
print("Results can be found in their respective output directories (output_RF0, output_RF1, etc.)")
print("Use analyze_results.py to generate comparison plots and tables.")
