import os
import traci
import numpy as np
import pandas as pd
from sumolib import checkBinary

def generate_simple_routes():
    """Generate simple routes using only valid edges from the network"""
    route_content = """<?xml version="1.0" encoding="UTF-8"?>
<routes>
    <vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="5" minGap="2.5" maxSpeed="17.88" guiShape="passenger"/>
    
    <!-- Simple routes using only valid edges -->
    <route id="route1" edges="-19448704#0 -19448704#1 -19448698"/>
    <route id="route2" edges="-19448298 -19448306#0 -19448306#1"/>
    <route id="route3" edges="-1347512023#0 -1347512023#1 -1347512024"/>
    
    <!-- Generate random traffic flows -->
    <flow id="f_route1" begin="0" end="3600" vehsPerHour="30" route="route1" type="car"/>
    <flow id="f_route2" begin="0" end="3600" vehsPerHour="25" route="route2" type="car"/>
    <flow id="f_route3" begin="0" end="3600" vehsPerHour="20" route="route3" type="car"/>
</routes>"""
    
    route_file = "simple_routes.rou.xml"
    with open(route_file, 'w') as f:
        f.write(route_content)
    
    return route_file

def create_temp_sumocfg(route_file):
    """Create a temporary SUMO config file that uses our route file"""
    config_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<sumoConfiguration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">

    <input>
        <net-file value="osm.net.xml.gz"/>
        <route-files value="{route_file}"/>
    </input>

    <processing>
        <ignore-route-errors value="true"/>
        <tls.actuated.jam-threshold value="30"/>
    </processing>

    <routing>
        <device.rerouting.adaptation-steps value="18"/>
        <device.rerouting.adaptation-interval value="10"/>
    </routing>

    <report>
        <verbose value="false"/>
        <duration-log.statistics value="true"/>
        <no-step-log value="true"/>
    </report>

</sumoConfiguration>"""
    
    temp_config = "temp_baseline.sumocfg"
    with open(temp_config, 'w') as f:
        f.write(config_content)
    
    return temp_config

# === Settings ===
num_episodes = 300
max_steps = 3600
interval = 30
output_csv = "simple_baseline_results.csv"

all_episode_results = []

for episode in range(num_episodes):
    print(f"Running Episode {episode+1}/{num_episodes}")
    
    # Generate routes and create temp config
    route_file = generate_simple_routes()
    temp_config = create_temp_sumocfg(route_file)
    
    if not route_file:
        print("Skipping episode due to route generation failure.")
        continue

    # Start SUMO directly without TrafficNetwork class
    sumo_binary = checkBinary("sumo")
    traci.start([sumo_binary, "-c", temp_config, "--no-step-log", "true", "--no-warnings", "true"])

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
            
            # Set all traffic lights to phase 0 (fixed timing)
            for tl_id in traci.trafficlight.getIDList():
                try:
                    traci.trafficlight.setPhase(tl_id, 0)
                except traci.TraCIException:
                    continue

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

                # Monitor all edges
                for edge_id in traci.edge.getIDList():
                    try:
                        edge_veh_ids = traci.edge.getLastStepVehicleIDs(edge_id)
                        for v in edge_veh_ids:
                            speed = traci.vehicle.getSpeed(v)
                            total_speed += speed
                            total_waiting += traci.vehicle.getWaitingTime(v)
                            total_vehicles += 1
                            if speed <= 0.1:
                                total_stopped += 1
                            else:
                                moving_vehicles += 1
                        total_queue += traci.edge.getLastStepHaltingNumber(edge_id)
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
        
    # Clean up temp files
    try:
        os.remove(temp_config)
        os.remove(route_file)
    except:
        pass

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

df = pd.DataFrame(all_episode_results)
df.to_csv(output_csv, index=False)
print(f"Saved all results to {output_csv}") 