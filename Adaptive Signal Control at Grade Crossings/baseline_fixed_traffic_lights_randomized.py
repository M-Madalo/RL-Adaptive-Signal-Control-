import os
import sys
import time
import traci
import numpy as np
import pandas as pd
import random

# === Ensure SUMO_HOME is set ===
if 'SUMO_HOME' not in os.environ:
    os.environ['SUMO_HOME'] = "C:/Program Files (x86)/Eclipse/Sumo"  # Adjust if needed
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))

# === Custom traffic generation function with seed control ===
def generate_traffic_with_seed(seed):
    """Generate traffic with controlled randomization using a specific seed - EXACT SAME as RL training"""
    np.random.seed(seed)
    random.seed(seed)
    
    # EXACT SAME flows as osm.rou.xml (all 42 flows)
    flows_data = [
        # (flow_id, from_edge, to_edge, via_edges, rate, is_number=False)
        ("f_0", "634020673#0", "-340808718#0", "E0 E0.79 1346606197 245915228#0 245915228#1 245915228#2 245915228#3 245915228#4 -1346338965 -1346082738#3 -1346082738#2 -1346082738#1 E1 E1.78 -1347512024 -1347512023#1 -1347512023#0 -1346082739 -340808718#10 -340808718#100 -340808718#9 -340808718#8 -340808718#7 -340808718#6 -340808718#5 -340808718#4 -340808718#3 -340808718#2 -340808718#1", 300),
        ("f_10", "-19470790#0", "1347512025", "935204071#1", 10),
        ("f_11", "-19470790#0", "19471592#2", "935204071#1 -E1 1346082738#1", 15),
        ("f_12", "-1347512025", "-340808718#0", "-1347512024 -1347512023#1 -1347512023#0 -1346082739 -340808718#10 -340808718#100 -340808718#9 -340808718#8 -340808718#7 -340808718#6 -340808718#5 -340808718#4 -340808718#3 -340808718#2 -340808718#1", 25),
        ("f_13", "-1347512025", "-634020673#0", "-E1 1346082738#1 1346082738#2 1346082738#3 1346338965 -245915228#4 -245915228#3 -245915228#2 -245915228#1 -245915228#0 -1346606197 -E0 -E00", 25),
        ("f_14", "-1347512025", "19470790#0", "-935204071#1", 25),
        ("f_15", "-19470790#0", "1347512025", "935204071#1", 25),
        ("f_16", "-344605828#0", "-340808718#0", "-1346338965 -1346082738#3 -1346082738#2 -1346082738#1 E1 E1.78 -1347512024 -1347512023#1 -1347512023#0 -1346082739 -340808718#10 -340808718#100 -340808718#9 -340808718#8 -340808718#7 -340808718#6 -340808718#5 -340808718#4 -340808718#3 -340808718#2 -340808718#1", 45),
        ("f_17", "-344605828#0", "19483852", "-245915228#4", 5),
        ("f_18", "-344605828#0", "-107195116#1", "", 25),
        ("f_19", "107195116#0", "634020683", "-1346338965 -1346082738#3 -1346082738#2 -1346082738#1 E1 E1.78 -1347512024 -1347512023#1 -1347512023#0 -1346082739 -340808718#10", 25),
        ("f_2", "340808718#0", "-634020673#0", "340808718#1 340808718#2 340808718#3 340808718#4 340808718#5 340808718#6 340808718#7 340808718#8 340808718#9 340808718#10 340808718#10.50 1346082739 1347512023#0 1347512023#1 1347512024 -E1 1346082738#1 1346082738#2 1346082738#3 1346338965 -245915228#4 -245915228#3 -245915228#2 -245915228#1 -245915228#0 -1346606197 -E0 -E00", 250, True),  # Uses number instead of vehsPerHour
        ("f_20", "107195116#0", "107193983", "-245915228#4 -245915228#3 -245915228#2 -245915228#1 -245915228#0 -1346606197", 50),
        ("f_21", "634020673#0", "1347512025", "245915228#4 E1.78", 50),
        ("f_22", "340808718#0", "-935204071#1", "340808718#10.50 1347512024", 50),
        ("f_23", "107195116#0", "344605828#0", "", 50),
        ("f_24", "-344605828#0", "-107195116#1", "", 50),
        ("f_25", "634020673#0", "-107195116#1", "E0 E0.79 1346606197 245915228#4", 50),
        ("f_26", "-19482366", "-107195116#1", "1346082738#3 1346338965", 5),
        ("f_27", "19471592#1", "19471592#2", "", 25),
        ("f_28", "19471592#1", "1347512025", "-1346082738#1 E1 E1.78", 25),
        ("f_29", "19471592#1", "-107195116#1", "1346082738#2 1346082738#3 1346338965", 50),
        ("f_3", "340808718#0", "-901681594#0", "340808718#1 340808718#2 340808718#3 340808718#4 340808718#5 340808718#6 340808718#7 340808718#8 340808718#9 340808718#10 340808718#10.50 -683562137#2", 30),
        ("f_30", "499316013#0", "-107195116#1", "245915228#1 245915228#2 245915228#3 245915228#4", 50),
        ("f_31", "-19448698", "-683562137#2", "340808718#9 340808718#10 340808718#10.50", 50),
        ("f_32", "-1347512025", "634020683", "-1347512024 -1347512023#1 -1347512023#0 -1346082739", 50),
        ("f_33", "107195116#0", "19468172", "-245915228#4 -245915228#3 -245915228#2", 50),
        ("f_34", "-344605828#0", "19468172", "-245915228#4 -245915228#3 -245915228#2", 50),
        ("f_35", "-19468172", "-340808718#0", "245915228#2 245915228#3 245915228#4 -1346338965 -1346082738#3 -1346082738#2 -1346082738#1 E1 E1.78 -1347512024 -1347512023#1 -1347512023#0 -1346082739 -340808718#10 -340808718#100 -340808718#9 -340808718#8 -340808718#7 -340808718#6 -340808718#5 -340808718#4 -340808718#3 -340808718#2 -340808718#1", 50),
        ("f_36", "634020673#0", "19468172", "E0 E0.79 1346606197 245915228#0 245915228#1", 75),
        ("f_37", "-19475676", "-19460106", "340808718#3 340808718#4 340808718#5 340808718#6 340808718#7 340808718#8 340808718#9 340808718#10 340808718#10.50 1346082739 1347512023#0 1347512023#1 1347512024 -E1 1346082738#1 1346082738#2 1346082738#3 1346338965 -245915228#4 -245915228#3 -245915228#2 -245915228#1 -245915228#0 -1346606197", 75),
        ("f_38", "19460106.847", "107193983", "", 50),
        ("f_39", "-107193983", "-634020673#0", "-E0 -E00", 75),
        ("f_4", "-634020683", "-19458273", "-683562137#2 -683562137#1 -683562137#0", 70),
        ("f_40", "-107193983", "-19460106", "", 60),
        ("f_41", "634020673#0", "107193983", "E0 E0.79", 40),
        ("f_5", "-634020683", "1347512025", "1346082739 1347512023#0 1347512023#1 1347512024", 25),
        ("f_6", "-634020683", "-340808718#0", "-340808718#10 -340808718#100 -340808718#9 -340808718#8 -340808718#7 -340808718#6", 25),
        ("f_7", "19448704#00", "634020683", "683562137#1 683562137#2", 25),
        ("f_8", "39447439#0", "634020683", "39447439#1 683562128 683562137#0 683562137#1 683562137#2", 25),
        ("f_9", "-19470790#0", "634020683", "935204071#1 -1347512024 -1347512023#1 -1347512023#0 -1346082739", 25),
    ]
    
    route_file_content = """<?xml version="1.0" encoding="UTF-8"?>
<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">
    <!-- VTypes -->
    <vType id="DEFAULT_RAILTYPE" length="2286.00" desiredMaxSpeed="8.94" vClass="rail"/>
    <vType id="DEFAULT_VEHTYPE" desiredMaxSpeed="17.88" color="yellow"/>
"""
    
    flows = []
    total_vehicles = 0
    
    # Add car flows with controlled randomization (±15% variation)
    for flow_id, from_edge, to_edge, via_edges, rate, *extra in flows_data:
        is_number = len(extra) > 0 and extra[0]
        
        if is_number:
            # f_2 uses number instead of vehsPerHour
            flow_rate = rate
            flow_content = f'    <flow id="{flow_id}" begin="0.00" color="blue" from="{from_edge}" to="{to_edge}"'
            if via_edges:
                flow_content += f' via="{via_edges}"'
            flow_content += f' end="3600.00" number="{flow_rate}"/>'
        else:
            # Apply controlled randomization with the provided seed
            variation = np.random.uniform(0.85, 1.15)  # ±15% variation
            flow_rate = int(rate * variation)
            flow_content = f'    <flow id="{flow_id}" begin="0.00" from="{from_edge}" to="{to_edge}"'
            if via_edges:
                flow_content += f' via="{via_edges}"'
            flow_content += f' end="3600.00" vehsPerHour="{flow_rate}"/>'
        
        flows.append((0.0, flow_content))
        total_vehicles += flow_rate
    
    # Add train flow (same as original)
    train_flow = '    <flow id="f_1" type="DEFAULT_RAILTYPE" begin="1200.00" from="832911536#0" to="832911536#1" end="3600.00" number="3"/>'
    flows.append((1200.0, train_flow))
    total_vehicles += 3
    
    print(f"Generated {total_vehicles} vehicles with seed {seed}")
    
    # Sort flows by departure time
    flows.sort(key=lambda x: x[0])
    
    # Add sorted flows to route file content
    for _, flow_content in flows:
        route_file_content += flow_content + '\n'
    
    route_file_content += "</routes>"
    
    # Write to temporary route file
    temp_route_file = f"temp_routes_episode_{seed}.rou.xml"
    with open(temp_route_file, "w") as f:
        f.write(route_file_content)
    
    return temp_route_file

# === Minimal inline version of TrafficNetwork with cleaned edges ===
class TrafficNetwork:
    def __init__(self, sumocfg_file, use_gui=False, route_file=None):
        self.sumocfg_file = sumocfg_file
        self.use_gui = use_gui
        self.route_file = route_file
        self._initialize_network()

    def _initialize_network(self):
        self.junctions = {
            "junction1": "202339061",
            "junction2": "202339032",
            "junction3": "202339043",
            "junction4": "202339017",
            "junction5": "202339039",
            "rail_crossing": "202291997"
        }
        self.monitored_edges = {
            "junction1": ["245915228#1", "-245915228#2", "-19468172"],
            "junction2": ["340808718#10.50", "683562137#2", "-1346082739", "-634020683"],
            "junction3": ["1347512024", "935204071#1", "-1347512025"],
            "junction4": ["1347512023#0", "-1347512023#1"],
            "junction5": ["1347512024", "935204071#1", "-1347512025"],
            "rail_crossing": ["245915228#1", "-245915228#2"]
        }

    def start_sumo(self):
        sumo_binary = "sumo-gui" if self.use_gui else "sumo"
        sumo_cmd = [sumo_binary, "-c", self.sumocfg_file, "--no-step-log", "true", "--no-warnings", "true"]
        
        # Add route file if specified
        if self.route_file:
            sumo_cmd.extend(["--route-files", self.route_file])
        
        traci.start(sumo_cmd)


# === Fixed-Timing Traffic Light Simulation (300 episodes) ===
sumocfg_file = "osm.sumocfg"
output_csv = "baseline_fixed_traffic_lights.csv"
num_episodes = 300
max_steps = 3600
interval = 30
all_episode_results = []

# Set initial random seed
base_seed = int(time.time() * 1000) % (2**32 - 1)
print(f"Base seed: {base_seed}")

for episode in range(num_episodes):
    # More robust randomization for each episode
    episode_seed = (base_seed + episode * 12345) % (2**32 - 1)
    
    # Add some entropy based on current time
    time.sleep(0.01)  # Small delay to ensure time differences
    additional_entropy = int(time.time() * 1000000) % 1000
    episode_seed = (episode_seed + additional_entropy) % (2**32 - 1)
    
    print(f"Running Episode {episode+1}/{num_episodes} (seed: {episode_seed})")
    
    # Generate traffic with the episode-specific seed
    route_file = generate_traffic_with_seed(episode_seed)
    if not route_file:
        print("Skipping episode due to traffic generation failure.")
        continue

    network = TrafficNetwork(sumocfg_file, use_gui=False, route_file=route_file)
    network.start_sumo()
    junctions = network.junctions
    monitored_edges = sum(network.monitored_edges.values(), [])

    vehicle_entry_time = {}
    total_travel_time = 0
    total_departed = 0
    waiting_times = []
    speeds = []
    queue_lengths = []
    stopped_vehicles = []

    for step in range(max_steps):
        traci.simulationStep()

        # Apply fixed phase 0 to all junctions
        for junction_id in junctions.values():
            try:
                traci.trafficlight.setPhase(junction_id, 0)
            except traci.TraCIException:
                continue

        current_time = traci.simulation.getTime()
        current_vehicles = traci.vehicle.getIDList()
        for vid in current_vehicles:
            if vid not in vehicle_entry_time:
                vehicle_entry_time[vid] = current_time

        departed = [vid for vid in list(vehicle_entry_time.keys()) if vid not in current_vehicles]
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

    traci.close()
    avg_travel_time = total_travel_time / max(total_departed, 1)

    result = {
        "episode": episode + 1,
        "avg_waiting_time": np.mean(waiting_times),
        "avg_speed": np.mean(speeds) * 2.237,  # Convert m/s to mph
        "avg_queue_length": np.mean(queue_lengths),
        "avg_stopped_vehicles": np.mean(stopped_vehicles),
        "avg_travel_time": avg_travel_time
    }
    all_episode_results.append(result)

    print(f"  Episode {episode+1} done - Avg Speed: {result['avg_speed']:.2f} mph, Avg Wait: {result['avg_waiting_time']:.2f} s, Avg Queue: {result['avg_queue_length']:.2f}, Avg Stopped: {result['avg_stopped_vehicles']:.2f}, Avg Travel Time: {result['avg_travel_time']:.2f} s")

df = pd.DataFrame(all_episode_results)
df.to_csv(output_csv, index=False)
print(f"Saved full baseline results to: {output_csv}")
