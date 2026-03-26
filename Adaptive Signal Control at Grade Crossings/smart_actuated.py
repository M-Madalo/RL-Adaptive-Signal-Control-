import traci
import numpy as np
import pandas as pd
import os
from collections import defaultdict

# Constants
MAX_GREEN_TIME = 45
MIN_GREEN_TIME = 5
COLLECTION_INTERVAL = 10
STOPPED_THRESHOLD = 0.1

# Replace these with your real junction and edge mappings
junctions = {
    "junction1": "202339061",
    "junction2": "202339032",
    "junction3": "202339043",
    "junction4": "202339017",
    "junction5": "202339039",
    "rail_crossing": "202291997"
}

monitored_edges = {
    "junction1": ["245915228#1", "-245915228#2", "-19468172"],
    "junction2": ["340808718#10.50", "683562137#2", "-1346082739", "-634020683"],
    "junction3": ["1347512024", "935204071#1", "-1347512025"],
    "junction4": ["1347512023#0", "-1347512023#1"],
    "junction5": ["1347512024", "935204071#1", "-1347512025"],
    "rail_crossing": ["245915228#1", "-245915228#2"]
}

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

def get_phase_pressures(junction_id, edge_ids):
    pressures = []
    logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(junction_id)[0]
    for phase_index, phase in enumerate(logic.phases):
        inflow = 0
        for edge in edge_ids:
            try:
                inflow += traci.edge.getLastStepHaltingNumber(edge)
            except:
                continue
        pressures.append((phase_index, inflow))
    return sorted(pressures, key=lambda x: -x[1])

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
                halting = traci.edge.getLastStepHaltingNumber(edge)
                lanes = traci.edge.getLaneNumber(edge)
                total_queue += halting / max(lanes, 1)
                for vid in vehs:
                    speed = traci.vehicle.getSpeed(vid)
                    if speed <= STOPPED_THRESHOLD:
                        stopped_vehicles += 1
                    else:
                        total_speed += speed * 2.237
                        speed_count += 1
                    total_waiting += traci.vehicle.getWaitingTime(vid)
            except:
                continue

    avg_waiting = total_waiting / max(stopped_vehicles, 1)
    avg_speed = total_speed / max(speed_count, 1)
    return avg_waiting, avg_speed, total_queue, stopped_vehicles

def smart_actuated_control(sim_time=3600):
    global travel_time_sum, vehicles_exited
    traci.start(["sumo", "-c", "osm.sumocfg"])
    step = 0
    current_phase = {j: 0 for j in junctions}
    green_time = {j: 0 for j in junctions}

    while step < sim_time:
        for junc, lid in junctions.items():
            edge_ids = monitored_edges[junc]
            phase_pressures = get_phase_pressures(lid, edge_ids)
            highest_pressure_phase = phase_pressures[0][0]

            if green_time[junc] >= MAX_GREEN_TIME or \
               (green_time[junc] >= MIN_GREEN_TIME and current_phase[junc] != highest_pressure_phase):
                traci.trafficlight.setPhase(lid, highest_pressure_phase)
                current_phase[junc] = highest_pressure_phase
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

def save_metrics_to_csv(metrics, path="smart_actuated_metrics.csv"):
    df = pd.DataFrame(metrics)
    df.to_csv(path, index=False)
    print(f"Saved metrics to: {path}")

if __name__ == "__main__":
    result = smart_actuated_control(sim_time=3600)
    save_metrics_to_csv(result)
