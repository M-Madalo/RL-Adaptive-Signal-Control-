import os
import sys
import time
import numpy as np
import pandas as pd
import traci
import matplotlib.pyplot as plt
import matplotlib as mpl

# Check SUMO environment
if 'SUMO_HOME' not in os.environ:
    sys.exit("Please declare environment variable 'SUMO_HOME'")
tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
sys.path.append(tools)

# Simulation configuration
script_dir = os.path.dirname(os.path.abspath(__file__))
sumocfg_file = os.path.join(script_dir, "osm.sumocfg")
sumo_cmd = ["sumo", "-c", sumocfg_file]

# Junction and signal timing configuration
junctions = {
    "junction1": "202339061",
    "junction2": "202339032",
    "junction3": "202339043",
    "junction4": "202339017",
    "junction5": "202339039",
    "rail_crossing": "202291997"
}

phases_config = {
    j: [31, 4, 6, 4] if j != "rail_crossing" else [60, 5, 30, 5] for j in junctions
}

monitored_edges = {
    "junction1": ["245915228#1", "-245915228#2", "-19468172"],
    "junction2": ["340808718#10.50", "683562137#2", "-1346082739", "-634020683"],
    "junction3": ["1347512024", "935204071#1", "-1347512025"],
    "junction4": ["1347512023#0", "-1347512023#1"],
    "junction5": ["1347512024", "935204071#1", "-1347512025"],
    "rail_crossing": ["245915228#1", "-245915228#2"]
}

# Metrics containers
metrics = {}
vehicle_start_times = {}
travel_time_sum = 0
vehicles_exited = 0
STOPPED_THRESHOLD = 0.1
COLLECTION_INTERVAL = 10
QUEUE_THRESHOLD = 5
MAX_GREEN_TIME = 30
MIN_GREEN_TIME = 5


def reset_metrics():
    global metrics, vehicle_start_times, travel_time_sum, vehicles_exited
    metrics = {
        'waiting_times': [],
        'avg_speeds': [],
        'queue_lengths': [],
        'travel_times': [],
        'stopped_vehicles': []
    }
    vehicle_start_times = {}
    travel_time_sum = 0
    vehicles_exited = 0


def collect_metrics():
    total_waiting = 0
    total_speed = 0
    speed_count = 0
    total_queue = 0
    stopped_vehicles = 0

    for edges in monitored_edges.values():
        for edge in edges:
            try:
                vehs = traci.edge.getLastStepVehicleIDs(edge)
                total_queue += traci.edge.getLastStepHaltingNumber(edge)
                for vid in vehs:
                    speed = traci.vehicle.getSpeed(vid)
                    if speed <= STOPPED_THRESHOLD:
                        stopped_vehicles += 1
                    else:
                        total_speed += speed * 2.237  # Convert m/s to mph
                        speed_count += 1
                    total_waiting += traci.vehicle.getWaitingTime(vid)
            except traci.TraCIException:
                continue

    avg_waiting = total_waiting / max(stopped_vehicles, 1)
    avg_speed = total_speed / max(speed_count, 1)
    return avg_waiting, avg_speed, total_queue, stopped_vehicles


def run_fixed_signal_simulation(sim_time=1800):
    global travel_time_sum, vehicles_exited
    traci.start(sumo_cmd)
    step = 0
    current_phase = {j: 0 for j in junctions}
    phase_timer = {j: phases_config[j][0] for j in junctions}

    while step < sim_time:
        for junc, lid in junctions.items():
            durations = phases_config[junc]
            if phase_timer[junc] == 0:
                current_phase[junc] = (current_phase[junc] + 1) % len(durations)
                traci.trafficlight.setPhase(lid, current_phase[junc])
                phase_timer[junc] = durations[current_phase[junc]]
            else:
                phase_timer[junc] -= 1

        traci.simulationStep()
        step += 1

        for vid in traci.vehicle.getIDList():
            if vid not in vehicle_start_times:
                vehicle_start_times[vid] = step

        exited = [vid for vid in list(vehicle_start_times) if vid not in traci.vehicle.getIDList()]
        for vid in exited:
            travel_time_sum += (step - vehicle_start_times[vid])
            vehicles_exited += 1
            del vehicle_start_times[vid]

        if step % COLLECTION_INTERVAL == 0:
            waiting, speed, queue, stopped = collect_metrics()
            metrics['waiting_times'].append(waiting)
            metrics['avg_speeds'].append(speed)
            metrics['queue_lengths'].append(queue)
            metrics['stopped_vehicles'].append(stopped)
            metrics['travel_times'].append(travel_time_sum / max(vehicles_exited, 1))

    traci.close()
    return metrics


def run_actuated_signal_simulation(sim_time=3600):
    global travel_time_sum, vehicles_exited
    traci.start(sumo_cmd)
    step = 0
    current_phase = {j: 0 for j in junctions}
    green_time = {j: 0 for j in junctions}

    while step < sim_time:
        for junc, lid in junctions.items():
            edges = monitored_edges[junc]
            queue_sum = 0
            for e in edges:
                try:
                    queue_sum += traci.edge.getLastStepHaltingNumber(e)
                except traci.TraCIException:
                    continue

            num_phases = len(traci.trafficlight.getCompleteRedYellowGreenDefinition(lid)[0].phases)
            if green_time[junc] >= MAX_GREEN_TIME or (green_time[junc] >= MIN_GREEN_TIME and queue_sum < QUEUE_THRESHOLD):
                current_phase[junc] = (current_phase[junc] + 1) % num_phases
                traci.trafficlight.setPhase(lid, current_phase[junc])
                green_time[junc] = 0
            else:
                green_time[junc] += 1

        traci.simulationStep()
        step += 1

        for vid in traci.vehicle.getIDList():
            if vid not in vehicle_start_times:
                vehicle_start_times[vid] = step

        exited = [vid for vid in list(vehicle_start_times) if vid not in traci.vehicle.getIDList()]
        for vid in exited:
            travel_time_sum += (step - vehicle_start_times[vid])
            vehicles_exited += 1
            del vehicle_start_times[vid]

        if step % COLLECTION_INTERVAL == 0:
            waiting, speed, queue, stopped = collect_metrics()
            metrics['waiting_times'].append(waiting)
            metrics['avg_speeds'].append(speed)
            metrics['queue_lengths'].append(queue)
            metrics['stopped_vehicles'].append(stopped)
            metrics['travel_times'].append(travel_time_sum / max(vehicles_exited, 1))

    traci.close()
    return metrics


def save_metrics_to_csv(metrics, path):
    df = pd.DataFrame(metrics)
    df.to_csv(path, index=False)
    print(f"Saved metrics to: {path}")


def plot_comparison(fixed_csv, actuated_csv):
    fixed = pd.read_csv(fixed_csv)
    actuated = pd.read_csv(actuated_csv)

    mpl.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "lines.linewidth": 1.5,
        "figure.figsize": (6, 4),
        "grid.color": "gray",
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
    })

    output_dir = "comparison_plots"
    os.makedirs(output_dir, exist_ok=True)

    units = {
        'avg_speeds': 'mph',
        'stopped_vehicles': 'Counts',
        'queue_lengths': 'Vehicles/lane',
        'travel_times': 'Seconds',
        'waiting_times': 'Seconds',
    }

    for column in fixed.columns:
        plt.figure()
        x_fixed = np.arange(len(fixed[column])) * COLLECTION_INTERVAL
        x_act = np.arange(len(actuated[column])) * COLLECTION_INTERVAL
        plt.plot(x_fixed, fixed[column], label="Fixed-Time", marker='o', markevery=max(len(fixed)//20, 1))
        plt.plot(x_act, actuated[column], label="Actuated", marker='s', markevery=max(len(actuated)//20, 1))

        plt.xlabel("Simulation Step")
        y_label = column.replace("_", " ").capitalize()
        if column in units:
            y_label += f" ({units[column]})"
        plt.ylabel(y_label)
        plt.title(f"{y_label} Comparison")
        plt.legend(loc="best", frameon=False)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{column}_comparison.png"), dpi=300, bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    print("Running Fixed-Time Signal Simulation for 3600 steps...")
    reset_metrics()
    result_fixed = run_fixed_signal_simulation(sim_time=3600)
    save_metrics_to_csv(result_fixed, "fts_metrics.csv")

    print("Running Actuated Signal Simulation for 3600 steps...")
    reset_metrics()
    result_actuated = run_actuated_signal_simulation(sim_time=3600)
    save_metrics_to_csv(result_actuated, "actuated_metrics.csv")

    print("Generating Comparison Plots...")
    plot_comparison("fts_metrics.csv", "actuated_metrics.csv")

    # Compute and save averages for each metric as CSV
    avg_fixed = {k: np.mean(v) for k, v in result_fixed.items()}
    avg_actuated = {k: np.mean(v) for k, v in result_actuated.items()}
    avg_df = pd.DataFrame({
        'Metric': list(avg_fixed.keys()),
        'Fixed-Time Average': [avg_fixed[k] for k in avg_fixed],
        'Actuated Average': [avg_actuated[k] for k in avg_fixed]
    })
    avg_df.to_csv('strategy_averages.csv', index=False)
    print("Saved strategy averages to: strategy_averages.csv")
