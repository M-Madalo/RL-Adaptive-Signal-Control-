import os
import traci
import numpy as np
import pandas as pd
from sumolib import checkBinary
from reinforcement_learningCAXX import generate_new_random_traffic
from traffic_network import TrafficNetwork

# === Settings (same as RL training) ===
sumocfg_file = "osm.sumocfg"
num_episodes = 300
max_steps = 3600
interval = 30
output_csv = "baseline_fixed_traffic_lights.csv"

all_episode_results = []

for episode in range(num_episodes):
    print(f"Running Episode {episode+1}/{num_episodes}")
    
    # Use the SAME traffic generation as RL training
    route_file = generate_new_random_traffic()
    if not route_file:
        print("Skipping episode due to route generation failure.")
        continue

    # Use the SAME network setup as RL training
    network = TrafficNetwork(sumocfg_file, use_gui=False)
    network.start_sumo()
    junctions = network.junctions
    monitored_edges = sum(network.monitored_edges.values(), [])

    step = 0
    waiting_times = []
    speeds = []
    queue_lengths = []
    stopped_vehicles = []
    vehicle_entry_time = {}
    total_travel_time = 0
    total_departed = 0

    try:
        while step < max_steps:
            traci.simulationStep()
            
            # FIXED TRAFFIC LIGHTS (no training/adaptation)
            # Set all traffic lights to phase 0 (fixed timing)
            for junc_name, junc_id in junctions.items():
                try:
                    traci.trafficlight.setPhase(junc_id, 0)  # Fixed phase 0
                except traci.TraCIException:
                    continue

            # SAME metrics collection as RL training
            current_time = traci.simulation.getTime()
            veh_ids = traci.vehicle.getIDList()
            for vid in veh_ids:
                if vid not in vehicle_entry_time:
                    vehicle_entry_time[vid] = current_time

            departed = [vid for vid in list(vehicle_entry_time.keys()) if vid not in veh_ids]
            for vid in departed:
                travel = current_time - vehicle_entry_time[vid]
                total_travel_time += travel
                total_departed += 1
                del vehicle_entry_time[vid]

            if step % interval == 0:
                total_waiting = 0
                total_speed = 0
                total_stopped = 0
                total_queue = 0
                total_vehicles = 0
                moving_vehicles = 0
                total_lanes = 0

                # SAME edge monitoring as RL training
                for edge in monitored_edges:
                    try:
                        edge_veh_ids = traci.edge.getLastStepVehicleIDs(edge)
                        for v in edge_veh_ids:
                            speed = traci.vehicle.getSpeed(v)
                            total_speed += speed
                            total_waiting += traci.vehicle.getWaitingTime(v)
                            total_vehicles += 1
                            if speed <= 0.1:
                                total_stopped += 1
                            else:
                                moving_vehicles += 1
                        total_queue += traci.edge.getLastStepHaltingNumber(edge)
                        total_lanes += 1
                    except traci.TraCIException:
                        continue

                avg_wait = total_waiting / max(total_vehicles, 1)
                avg_speed = total_speed / max(moving_vehicles, 1)
                avg_queue = total_queue / max(total_lanes, 1)

                waiting_times.append(avg_wait)
                speeds.append(avg_speed)
                queue_lengths.append(avg_queue)
                stopped_vehicles.append(total_stopped)

            step += 1
    finally:
        traci.close()

    # SAME result calculation as RL training
    avg_travel_time = total_travel_time / max(total_departed, 1)

    result = {
        "episode": episode + 1,
        "avg_waiting_time": np.mean(waiting_times),
        "avg_speed": np.mean(speeds),
        "avg_queue_length": np.mean(queue_lengths),
        "avg_stopped_vehicles": np.mean(stopped_vehicles),
        "avg_travel_time": avg_travel_time
    }
    all_episode_results.append(result)
    
    print(f"  Episode {episode+1} completed - Avg Speed: {result['avg_speed']:.2f} m/s, Avg Wait: {result['avg_waiting_time']:.2f} s")

# Save results in SAME format as RL training
df = pd.DataFrame(all_episode_results)
df.to_csv(output_csv, index=False)
print(f"Saved baseline results to {output_csv}")
print(f"Baseline simulation completed with FIXED traffic lights (no training)") 