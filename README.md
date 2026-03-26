# RL-Adaptive-Signal-Control-
This framework is a corridor-level adaptive traffic signal control system designed for urban networks affected by railroad-highway grade crossings (RHGCs). It combines a SUMO-based simulation environment with a Deep Q-Network (DQN) agent  that observes multimodal traffic conditions  and selects signal actions to reduce congestion.

# Signal Optimization in Multimodal Corridors with Railroad-Highway Grade Crossings

This repository contains the simulation and reinforcement learning workflow used for the paper:

**"Signal Optimization in Multimodal Corridors with Railroad-Highway Grade Crossings"**

The project uses SUMO + TraCI + Deep Reinforcement Learning (DQN) to optimize signal control in a multimodal corridor impacted by rail crossing events.

## Short Framework Description

This framework is a corridor-level adaptive traffic signal control system designed for urban networks affected by railroad-highway grade crossings (RHGCs). It combines a SUMO-based simulation environment with a Deep Q-Network (DQN) agent that observes multimodal traffic conditions, including vehicle flow and train disruptions, and selects signal actions to reduce congestion and improve operational stability. The framework supports reward-function benchmarking (RF0-RF4), fixed-time baseline comparison, and reproducible performance analysis using delay, queue length, speed, travel time, and stopped-vehicle metrics.

## Project Scope

- Corridor-level adaptive signal control near a railroad-highway grade crossing (RHGC)
- DQN-based agent with reward variants `RF0` to `RF4`
- Randomized traffic demand across episodes
- Fixed-time baseline for comparison
- Post-processing scripts for result tables and figures

## Repository Structure

### Core Training and Evaluation

- `reinforcement_learningCAXX.py` - main DRL training script
- `train_all_variants.py` - runs all reward variants (`RF0` to `RF4`)
- `baseline_fixed_traffic_lights_randomized.py` - fixed-time baseline simulation
- `analyze_results.py` - aggregates and visualizes training metrics
- `compare_baseline_rf2.py` - compares RF2 against fixed-time baseline

### SUMO Network and Config

- `osm.sumocfg` - SUMO scenario configuration
- `osm.rou.xml` - base route definitions
- `osm.net.xml.gz` - network file
- `osm.netccfg`, `osm.view.xml` - SUMO support configs

### Generated Outputs

- `output/` - per-variant training metrics (e.g., `output/output_RF2/RF2training_metrics.csv`)
- `figures/` - summary tables and figures
- `baseline_fixed_traffic_lights.csv` - baseline metrics
- `baseline_vs_rf2_comparison.csv` - comparison table

## Requirements

### 1) Software

- Python 3.9+ (recommended 3.10+)
- [SUMO](https://www.eclipse.org/sumo/) installed locally

### 2) Python Packages

Install common dependencies:

```bash
pip install numpy pandas matplotlib seaborn scipy torch tensorboard
```

### 3) Environment Variable

Set `SUMO_HOME` to your SUMO installation directory and ensure `tools` is available.

Example (Windows PowerShell):

```powershell
$env:SUMO_HOME="C:\Program Files (x86)\Eclipse\Sumo"
```

## How to Run

### A. Train All Reward Variants (RF0-RF4)

```bash
python train_all_variants.py
```

This launches `reinforcement_learningCAXX.py` repeatedly for each reward mode and stores results in `output/output_RF*/`.

### B. Train a Single Variant (optional)

```bash
python reinforcement_learningCAXX.py --episodes 300 --reward_mode RF2
```

### C. Run Fixed-Time Baseline

```bash
python baseline_fixed_traffic_lights_randomized.py
```

### D. Generate Summary Tables and Figures

```bash
python analyze_results.py
```

### E. Compare Baseline vs RF2

```bash
python compare_baseline_rf2.py
```

## Expected Outputs

After a standard run, you should have:

- `output/output_RF0/RF0training_metrics.csv`
- `output/output_RF1/RF1training_metrics.csv`
- `output/output_RF2/RF2training_metrics.csv`
- `output/output_RF3/RF3training_metrics.csv`
- `output/output_RF4/RF4training_metrics.csv`
- `figures/final_results_table_detailed.csv`
- `baseline_fixed_traffic_lights.csv`
- `baseline_vs_rf2_comparison.csv`

## Reproducibility Notes

- Reward variants are evaluated under a consistent training protocol.
- `train_all_variants.py` is configured for `300` episodes per variant.
- Baseline and RL comparison should be run on the same network and demand assumptions.
- Temporary route files may be created during simulation runtime.

## Troubleshooting

- **SUMO_HOME not set**: set `SUMO_HOME` before running scripts.
- **TraCI import errors**: verify SUMO installation and Python path setup.
- **Missing output files**: ensure each training/baseline script completes successfully before running analysis scripts.
- **Long runtime**: training across all variants can take significant time depending on hardware.

## Citation

If you use this codebase, please cite your paper:

Signal Optimization in Multimodal Corridors with Railroad-Highway Grade Crossings.


