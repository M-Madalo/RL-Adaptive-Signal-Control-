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


def run_fixed_signal_simulation(sim_time=3600):
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


def save_metrics_to_csv(metrics, path="fts_metrics.csv"):
    df = pd.DataFrame(metrics)
    df.to_csv(path, index=False)
    print(f"Saved metrics to: {path}")


def plot_trb_compliant_metrics(metrics):
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

    # Create output directory if it doesn't exist
    plot_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ftsplots')
    os.makedirs(plot_dir, exist_ok=True)

    time_steps = [i * COLLECTION_INTERVAL for i in range(len(next(iter(metrics.values()))))]

    # Define units for each metric
    units = {
        'avg_speeds': 'mph',
        'stopped_vehicles': 'Counts',
        'queue_lengths': 'Vehicles/lane',
        'travel_times': 'Seconds',
        'waiting_times': 'Seconds',
    }

    for key, values in metrics.items():
        plt.figure()
        plt.plot(time_steps, values, label=key.replace("_", " ").capitalize(), marker='o', markevery=max(len(values)//20, 1))
        plt.xlabel("Simulation Step")
        # Set y-axis label with unit if available
        y_label = key.replace("_", " ").capitalize()
        if key in units:
            y_label += f" ({units[key]})"
        plt.ylabel(y_label)
        plt.title(f"{key.replace('_', ' ').capitalize()} Over Time")
        plt.legend(loc="best", frameon=False)
        plt.tight_layout()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        # Mark and annotate the average value
        avg_value = np.mean(values)
        mid_x = time_steps[len(time_steps)//2] if time_steps else 0
        plt.scatter([mid_x], [avg_value], color='red', zorder=5)
        plt.annotate(f"Avg: {avg_value:.2f}", (mid_x, avg_value), textcoords="offset points", xytext=(10,5), ha='left', color='red', fontsize=10, weight='bold')
        # Save plot to ftsplots directory
        filename = f"{key}.png"
        save_path = os.path.join(plot_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved plot: {save_path}")


if __name__ == "__main__":
    print("Running Fixed-Time Signal Simulation for 3600 steps...")
    reset_metrics()
    result = run_fixed_signal_simulation(sim_time=3600)
    save_metrics_to_csv(result)
    plot_trb_compliant_metrics(result)
