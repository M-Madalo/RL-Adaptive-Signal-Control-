import os
import sys
import time
import math
import random
import argparse
import signal
import atexit
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from collections import deque, namedtuple
import traci
import subprocess
import threading
import xml.etree.ElementTree as ET
import tempfile
import shutil
import json
import warnings
from typing import Dict, List, Tuple, Optional, Any
import logging

# Robust device import
try:
    from traffic_components import device
except ImportError:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

warnings.filterwarnings("ignore")

# Global variables for graceful shutdown
shutdown_requested = False
current_env = None

def signal_handler(signum, frame):
    """Handle interrupt signals gracefully"""
    global shutdown_requested, current_env
    print(f"\nReceived signal {signum}. Initiating graceful shutdown...")
    shutdown_requested = True
    
    # Clean up current environment if it exists
    if current_env is not None:
        try:
            current_env.cleanup()
            print("Environment cleaned up successfully.")
        except Exception as e:
            print(f"Warning: Error during environment cleanup: {e}")
    
    # Exit gracefully
    sys.exit(0)

def cleanup_on_exit():
    """Cleanup function called on normal exit"""
    global current_env
    if current_env is not None:
        try:
            current_env.cleanup()
        except Exception as e:
            print(f"Warning: Error during exit cleanup: {e}")

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
atexit.register(cleanup_on_exit)

# Define performance targets globally
performance_targets = {
    'waiting_time': {
        'excellent': 30,  # seconds
        'good': 60,      # seconds
        'critical': 90   # seconds
    },
    'lane_thresholds': {
        'excellent': 2,    # vehicles per lane
        'good': 3,        # vehicles per lane
        'critical': 5     # vehicles per lane (maximum allowed)
    },
    'speed': {
        'target': 40.0,   # Target speed in mph
        'excellent': 35.0,
        'good': 30.0,
        'poor': 20.0,
        'critical': 5.0
    }
}

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# We need to import Python modules from the $SUMO_HOME/tools directory
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

import traci
import sumolib

# SUMO Configuration
sumocfg_file = os.path.join(script_dir, "osm.sumocfg")
sumo_cmd = ["sumo", "-c", sumocfg_file]

print(f"SUMO configuration file path: {sumocfg_file}")
print(f"Current working directory: {os.getcwd()}")

# Check if the configuration file exists
if not os.path.exists(sumocfg_file):
    raise FileNotFoundError(f"SUMO configuration file not found: {sumocfg_file}")

def generate_new_random_traffic():
    """Generate a new route file with randomized traffic flows while maintaining train schedule.
    This ensures varied training conditions while keeping critical train patterns consistent.
    Traffic flows vary by ±15% to create different patterns each episode."""
    max_attempts = 10  # Maximum attempts to generate valid traffic
    min_vehicles = 100  # Minimum total vehicles required
    for attempt in range(max_attempts):
        try:
            routes = {
                "route1": ["19448704#00", "683562137#1", "683562137#2", "634020683"],
                "route2": ["39447439#0", "39447439#1", "683562128", "683562137#0", "683562137#1", "683562137#2", "634020683"],
                "route3": ["-19470790#0", "935204071#1", "-1347512024", "-1347512023#1", "-1347512023#0", "-1346082739"],
                "route4": ["245915228#1", "-245915228#2", "-19468172"],
                "route5": ["340808718#10.50", "683562137#2", "-1346082739", "-634020683"],
                "route6": ["634020673#0", "-340808718#0", "E0", "E0.79", "1346606197", "245915228#0", "245915228#1", "245915228#2", "245915228#3", "245915228#4", "-1346338965", "-1346082738#3", "-1346082738#2", "-1346082738#1", "E1", "E1.78", "-1347512024", "-1347512023#1", "-1347512023#0", "-1346082739", "-340808718#10", "-340808718#100", "-340808718#9", "-340808718#8", "-340808718#7", "-340808718#6", "-340808718#5", "-340808718#4", "-340808718#3", "-340808718#2", "-340808718#1"],
            }
            original_flows = {
                "f_0": 300, "f_2": 250, "f_3": 30, "f_4": 70, "f_5": 25, "f_6": 25,
                "f_7": 25, "f_8": 25, "f_9": 25, "f_10": 10, "f_11": 15, "f_12": 25,
                "f_13": 25, "f_14": 25, "f_15": 25, "f_16": 45, "f_17": 5, "f_18": 25,
                "f_19": 25, "f_20": 50, "f_21": 50, "f_22": 50, "f_23": 50, "f_24": 50,
                "f_25": 50, "f_26": 5, "f_27": 25, "f_28": 25, "f_29": 50, "f_30": 50,
                "f_31": 50, "f_32": 50, "f_33": 50, "f_34": 50, "f_35": 50, "f_36": 75,
                "f_37": 75, "f_38": 50, "f_39": 75, "f_40": 60, "f_41": 40
            }
            route_file_content = """<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<routes>\n    <!-- Vehicle Types -->\n    <vType id=\"car\" accel=\"2.6\" decel=\"4.5\" sigma=\"0.5\" length=\"5\" minGap=\"2.5\" maxSpeed=\"17.88\" guiShape=\"passenger\"/>\n    <vType id=\"DEFAULT_RAILTYPE\" length=\"2286.00\" desiredMaxSpeed=\"8.94\" vClass=\"rail\"/>\n"""
            flows = []
            total_vehicles = 0
            for route_id, edges in routes.items():
                via = " ".join(edges[1:-1]) if len(edges) > 2 else ""
                original_rate = original_flows.get(f"f_{route_id[-1]}", 25)
                if route_id == "route2":
                    flow_rate = 250
                else:
                    variation = np.random.uniform(0.85, 1.15)
                    flow_rate = int(original_rate * variation)
                vehicles_per_flow = int(flow_rate)
                total_vehicles += vehicles_per_flow
                flow_content = f'    <flow id="f_{route_id}" begin="0.00" from="{edges[0]}" to="{edges[-1]}"'
                if via:
                    flow_content += f' via="{via}"'
                flow_content += f' end="3600.00" vehsPerHour="{flow_rate}" type="car"/>'
                flows.append((0.0, flow_content))
            train_flows = [
                (1200.0, '    <flow id="f_train1" type="DEFAULT_RAILTYPE" begin="1200.00" from="832911536#0" to="832911536#1" end="1200.00" number="1"/>'),
                (2400.0, '    <flow id="f_train2" type="DEFAULT_RAILTYPE" begin="2400.00" from="832911536#0" to="832911536#1" end="2400.00" number="1"/>'),
                (3600.0, '    <flow id="f_train3" type="DEFAULT_RAILTYPE" begin="3600.00" from="832911536#0" to="832911536#1" end="3600.00" number="1"/>')
            ]
            flows.extend(train_flows)
            total_vehicles += 3
            if total_vehicles < min_vehicles:
                print(f"Attempt {attempt + 1}: Generated {total_vehicles} vehicles (below threshold of {min_vehicles}), regenerating...")
                continue
            print(f"Generated {total_vehicles} vehicles (above threshold of {min_vehicles})")
            flows.sort(key=lambda x: x[0])
            for _, flow_content in flows:
                route_file_content += flow_content + '\n'
            route_file_content += "</routes>"
            temp_route_file = "temp_routes.rou.xml"
            with open(temp_route_file, "w") as f:
                f.write(route_file_content)
            return temp_route_file
        except Exception as e:
            print(f"Error generating traffic (attempt {attempt + 1}): {e}")
            if attempt == max_attempts - 1:
                print(f"Failed to generate valid traffic after {max_attempts} attempts")
                return None
            continue
    print(f"Failed to generate traffic with minimum {min_vehicles} vehicles after {max_attempts} attempts")
    return None

# DQN-specific imports and setup


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

class SimpleReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)  # Use deque for efficient FIFO operations

    def push(self, *args):
        self.memory.append(Transition(*args))  # Store transitions

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)  # Randomly sample transitions

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self, state_size, hidden_size, junction_actions):
        super(DQN, self).__init__()
        self.junction_actions = junction_actions
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        
        self.output_heads = nn.ModuleDict()
        for junction, n_phases in junction_actions.items():
            self.output_heads[junction] = nn.ModuleDict({
                'phase': nn.Linear(hidden_size // 2, n_phases),
                'duration': nn.Linear(hidden_size // 2, 1)
        })
        
    def forward(self, x):
        if len(x.shape) == 2:
            x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        
        out = {}
        for junction, heads in self.output_heads.items():
            out[junction] = {
                'phase_logits': heads['phase'](x),
                'duration': heads['duration'](x)
            }
        return out

class TrafficEnvironment:
    def __init__(self, sumocfg_file, max_steps=3600, measurement_interval=10, use_gui=False, num_vehicles=50, reward_mode='baseline'):
        global current_env
        current_env = self
        
        self.sumocfg_file = sumocfg_file
        self.max_steps = max_steps
        self.measurement_interval = measurement_interval
        self.use_gui = use_gui
        self.num_vehicles = num_vehicles
        self.reward_mode = reward_mode
        
        # Connection management
        self.connection_initialized = False
        self.connection_closed = False
        
        # Check SUMO_HOME environment variable
        if 'SUMO_HOME' in os.environ:
            tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
            sys.path.append(tools)
        else:
            sys.exit("Please declare environment variable 'SUMO_HOME'")
        
        # Initialize other attributes
        self._initialize_network()
        self._initialize_metrics()
        
        # Performance tracking
        self.running_reward_baseline = 0.0
        
        # TensorBoard logging
        self.writer = SummaryWriter(f'runs/traffic_control_{int(time.time())}')
        
        print(f"Environment initialized with {len(self.controllable_junctions)} controllable junctions")
        print(f"Monitored edges: {len(self.monitored_edges)} junctions with {sum(len(edges) for edges in self.monitored_edges.values())} total edges")

    def reset(self):
        """Reset the environment for a new episode"""
        global shutdown_requested
        
        # Check if shutdown was requested
        if shutdown_requested:
            self.done = True
            return self._get_state()
        
        # Close existing SUMO connection if it exists
        self.cleanup()
        
        # Initialize new SUMO connection
        try:
            if self.use_gui:
                sumo_binary = "sumo-gui"
            else:
                sumo_binary = "sumo"
            
            # Start SUMO with the configuration file
            traci.start([sumo_binary, "-c", self.sumocfg_file, "--no-step-log", "true", "--no-warnings", "true"])
            self.connection_initialized = True
            self.connection_closed = False
            
            # Initialize episode-specific variables
            self.steps = 0
            self.done = False
            self.episode += 1
            
            # Reset metrics for new episode
            self._initialize_metrics()
            
            # Initialize traffic lights to default phases
            for junction_name in self.controllable_junctions:
                junction_id = self.junctions[junction_name]
                traci.trafficlight.setPhase(junction_id, 0)
                self.current_phases[junction_name] = 0
                self.last_phase_changes[junction_name] = 0
            
            return self._get_state()
            
        except Exception as e:
            print(f"Error initializing SUMO connection: {e}")
            self.done = True
            return self._get_state()

    def _initialize_network(self):
        """Initialize the traffic network configuration"""
        try:
            # Define junction IDs
            self.junctions = {
                "junction1": "202339061",  # Traffic light junction
                "junction2": "202339032",  # Traffic light junction
                "junction3": "202339043",  # Traffic light junction
                "junction4": "202339017",  # Traffic light junction
                "junction5": "202339039",  # Traffic light junction
                "rail_crossing": "202291997"  # Rail crossing junction
            }
            # Define monitored edges for each junction
            self.monitored_edges = {
                "junction1": ["245915228#1", "-245915228#2", "-19468172"],
                "junction2": ["340808718#10.50", "683562137#2", "-1346082739", "-634020683"],
                "junction3": ["1347512024", "935204071#1", "-1347512025"],
                "junction4": ["1347512023#0", "-1347512023#1"],
                "junction5": ["1347512024", "935204071#1", "-1347512025"],
                "rail_crossing": ["245915228#1", "-245915228#2"]
            }
            # Define controllable junctions with their phases
            self.controllable_junctions = {
                "junction1": {
                    "phases": [
                        "GGggrrrrGGGggrrrrr",
                        "yyggrrrryyyggrrrrr",
                        "rrGGrrrrrrrGGrrrrr",
                        "rryyrrrrrrryyrrrrr"
                    ],
                    "durations": [31, 4, 6, 4]
                },
                "junction2": {
                    "phases": [
                        "GGGggrrrrGGGggrrrr",
                        "yyyggrrrryyyggrrrr",
                        "rrrGGrrrrrrrGGrrrr",
                        "rrryyrrrrrrryyrrrr"
                    ],
                    "durations": [31, 4, 6, 4]
                },
                "junction3": {
                    "phases": [
                        "GGGggrrrrGGGggrrrr",
                        "yyyggrrrryyyggrrrr",
                        "rrrGGrrrrrrrGGrrrr",
                        "rrryyrrrrrrryyrrrr"
                    ],
                    "durations": [31, 4, 6, 4]
                },
                "junction4": {
                    "phases": [
                        "GGGggrrrrGGGggrrrr",
                        "yyyggrrrryyyggrrrr",
                        "rrrGGrrrrrrrGGrrrr",
                        "rrryyrrrrrrryyrrrr"
                    ],
                    "durations": [31, 4, 6, 4]
                },
                "junction5": {
                    "phases": [
                        "GGGggrrrrGGGggrrrr",
                        "yyyggrrrryyyggrrrr",
                        "rrrGGrrrrrrrGGrrrr",
                        "rrryyrrrrrrryyrrrr"
                    ],
                    "durations": [31, 4, 6, 4]
                },
                "rail_crossing": {
                    "phases": [
                        "GGGGGGGGGGGGGGGGGG",
                        "yyyyyyyyyyyyyyyyyy",
                        "rrrrrrrrrrrrrrrrrr",
                        "yyyyyyyyyyyyyyyyyy"
                    ],
                    "durations": [60, 5, 30, 5]
                }
            }
            # Initialize phases attribute
            self.phases = {j: self.controllable_junctions[j]["phases"] for j in self.controllable_junctions}
            # Initialize phase tracking
            self.current_phases = {j: 0 for j in self.controllable_junctions}
            self.last_phase_changes = {j: 0 for j in self.controllable_junctions}
            self.phase_durations = {j: {p: 0 for p in range(len(self.controllable_junctions[j]["phases"]))} for j in self.controllable_junctions}
            
            # Initialize rail-related attributes
            self.rail_edges = []
            self.rail_lookahead_edges = []
            self.train_approaching = False
            self.train_distance = float('inf')
            self.train_speed = 0.0
            self.train_safety_score = 0.0
            self.train_approaching_soon = False
            
            # Initialize thresholds
            self.queue_threshold = 5  # Maximum allowed queue length before forcing phase change
            self.min_speed_threshold = 5.0  # Minimum speed threshold in m/s
            self.speed_check_interval = 30  # Check speed every 30 steps
            self.min_green_time = 30  # Increased minimum green time from 20 to 30 seconds
            
            # Define state and action spaces
            self.state_size = 49  # Updated state size for additional intersections
            # Define action space based on traffic light phases
            self.junction_actions = {}
            for junction_name in self.controllable_junctions:
                self.junction_actions[junction_name] = len(self.controllable_junctions[junction_name]["phases"])
            self.action_size = 1
            for n in self.junction_actions.values():
                self.action_size *= n
                
            # Add convergence tracking
            self.convergence_window = 20
            self.convergence_threshold = 0.1
            self.reward_history = []
            self.best_reward = float('-inf')
            self.no_improvement_count = 0
            self.max_no_improvement = 25
            # Add episode metrics buffer
            self.episode_metrics_buffer = []
            
        except traci.exceptions.TraCIException as e:
            self.connection_initialized = False
            self.connection_closed = True
            raise
        except Exception as e:
            self.connection_initialized = False
            self.connection_closed = True
            raise

    def _check_train_presence(self):
        """Check for trains in the network and their status, with lookahead."""
        self.train_approaching_soon = False  # Reset lookahead flag
        for edge in self.rail_edges:
            vehicles = traci.edge.getLastStepVehicleIDs(edge)
            for v in vehicles:
                if traci.vehicle.getVehicleClass(v) == 'rail':
                    pos = traci.vehicle.getLanePosition(v)
                    speed = traci.vehicle.getSpeed(v)
                    self.train_approaching = True
                    self.train_distance = pos
                    self.train_speed = speed * 2.237  # Convert m/s to mph
                    return True
        # Lookahead: check upstream edges
        for edge in self.rail_lookahead_edges:
            vehicles = traci.edge.getLastStepVehicleIDs(edge)
            for v in vehicles:
                if traci.vehicle.getVehicleClass(v) == 'rail':
                    self.train_approaching_soon = True
                    break
        self.train_approaching = False
        self.train_distance = float('inf')
        self.train_speed = 0
        return False

    def _get_state(self):
        """Get enhanced state including train information and queue lengths"""
        # Base state size: 42 traffic features + 3 train features + 4 queue features
        state = np.zeros(49, dtype=np.float32)
        
        try:
            if not traci.isLoaded() or traci.simulation.getMinExpectedNumber() <= 0:
                return state
            
            # Update train status (only if needed)
            if self.steps % 5 == 0:  # Check train presence less frequently
                self._check_train_presence()
            
            # Add train features
            state[42] = float(self.train_approaching)
            state[43] = min(self.train_distance / 1000.0, 1.0)  # Normalize distance
            state[44] = min(self.train_speed / 10.0, 1.0)      # Normalize speed
            
            # Add traffic features for each junction
            idx = 0
            for junction_name, junction_id in self.junctions.items():
                edges = self.monitored_edges.get(junction_name, [])
                if not edges:
                    continue
                    
                for edge in edges[:4]:  # Limit to 4 edges per junction
                    try:
                        if idx < 42:  # Only fill up to index 41 for traffic features
                            vehicle_count = traci.edge.getLastStepVehicleNumber(edge)
                            waiting_time = traci.edge.getWaitingTime(edge)
                            mean_speed = traci.edge.getLastStepMeanSpeed(edge)
                            queue_length = traci.edge.getLastStepHaltingNumber(edge)
                            
                            state[idx] = min(vehicle_count / 20.0, 1.0)
                            state[idx + 1] = min(waiting_time / 300.0, 1.0)
                            state[idx + 2] = min(mean_speed / 17.88, 1.0)
                            state[idx + 3] = min(queue_length / 10.0, 1.0)
                    except traci.exceptions.TraCIException:
                        if idx < 42:
                            state[idx:idx+4] = [0.0, 0.0, 0.0, 0.0]
                    idx += 4
                
                # Fill remaining slots if less than 4 edges
                remaining_slots = 4 - min(len(edges), 4)
                if remaining_slots > 0 and idx < 42:
                    idx += remaining_slots * 4
            
            return state
            
        except traci.exceptions.FatalTraCIError:
            self.done = True
            return state
        except Exception:
            return state

    def _check_queues(self):
        """Check queue lengths at monitored edges"""
        total_queues = {}
        total_waiting_time = 0
        
        try:
            for junction_name, edges in self.monitored_edges.items():
                junction_queue = 0
                junction_waiting = 0
                
                for edge in edges:
                    try:
                        # Get number of vehicles and their waiting times
                        vehicles = traci.edge.getLastStepVehicleNumber(edge)
                        waiting_time = traci.edge.getWaitingTime(edge)
                        
                        # Consider a vehicle as queued if it has waiting time > 0
                        if waiting_time > 0:
                            junction_queue += vehicles
                            junction_waiting += waiting_time
                    except traci.exceptions.TraCIException:
                        continue
                
                total_queues[junction_name] = junction_queue
                total_waiting_time += junction_waiting
        
        except traci.exceptions.FatalTraCIError:
            self.done = True
            return {}, 0
            
        return total_queues, total_waiting_time

    def _update_travel_times(self):
        """Update traveling times for vehicles"""
        try:
            # Get all vehicles in the network
            all_vehicles = traci.vehicle.getIDList()
            
            # Track new vehicles
            for vehicle_id in all_vehicles:
                if vehicle_id not in self.vehicle_travel_times:
                    self.vehicle_travel_times[vehicle_id] = traci.simulation.getTime()
            
            # Check for vehicles that have left the network
            departed_vehicles = []
            for vehicle_id in self.vehicle_travel_times:
                if vehicle_id not in all_vehicles:
                    # Vehicle has left the network
                    start_time = self.vehicle_travel_times[vehicle_id]
                    end_time = traci.simulation.getTime()
                    travel_time = end_time - start_time
                    
                    self.total_travel_time += travel_time
                    self.total_vehicles_passed += 1
                    departed_vehicles.append(vehicle_id)
            
            # Remove departed vehicles from tracking
            for vehicle_id in departed_vehicles:
                del self.vehicle_travel_times[vehicle_id]
            
        except traci.exceptions.TraCIException as e:
            print(f"Warning: Error updating travel times: {e}")
        except Exception as e:
            print(f"Unexpected error updating travel times: {e}")

    def _get_average_travel_time(self):
        """Calculate average traveling time for vehicles that have passed"""
        if self.total_vehicles_passed == 0:
            return 0
        return self.total_travel_time / self.total_vehicles_passed

    def _measure_traffic_speed(self):
        """Measure traffic speed with proper speed categories"""
        total_speed = 0
        total_moving_vehicles = 0
        MIN_SPEED_THRESHOLD = 2.2352  # 5 mph in m/s (critical threshold)
        
        try:
            for junction_name, edges in self.monitored_edges.items():
                for edge in edges:
                    try:
                        vehicles = traci.edge.getLastStepVehicleIDs(edge)
                        for vid in vehicles:
                            speed = traci.vehicle.getSpeed(vid)
                            if speed > MIN_SPEED_THRESHOLD:
                                speed_mph = speed * 2.237  # Convert m/s to mph
                                total_speed += speed_mph
                                total_moving_vehicles += 1
                                
                                # Log per-vehicle speeds
                                self.writer.add_scalar(f'Junction_{junction_name}/Vehicle_{vid}/Speed', 
                                                     speed_mph, self.steps)
                    except traci.exceptions.TraCIException:
                        continue
            
            if total_moving_vehicles == 0:
                return 0.0
            
            avg_speed = total_speed / total_moving_vehicles
            
            # Log speed metrics
            self.writer.add_scalar('Metrics/Average_Speed', avg_speed, self.steps)
            self.writer.add_scalar('Metrics/Moving_Vehicles', total_moving_vehicles, self.steps)
            
            return avg_speed
        except Exception as e:
            print(f"Error measuring traffic speed: {e}")
            return 0.0

    def _get_traffic_stats(self):
        """Get detailed traffic statistics including stopped and waiting vehicles"""
        stats = {
            'total_vehicles': 0,
            'moving_vehicles': 0,
            'stopped_vehicles': 0,  # Completely stopped (speed = 0)
            'waiting_vehicles': 0,  # Either stopped or moving very slowly
            'avg_speed_moving': 0.0,
            'avg_speed_all': 0.0
        }
        
        try:
            total_speed_moving = 0
            total_speed_all = 0
            STOPPED_THRESHOLD = 0.1  # Consider vehicles moving slower than this as "stopped"
            WAITING_THRESHOLD = 0.3  # Lowered from 0.5 to 0.3 m/s to better filter actual waiting vs crawling
            
            for junction_name, edges in self.monitored_edges.items():
                for edge in edges:
                    try:
                        vehicles = traci.edge.getLastStepVehicleIDs(edge)
                        for vid in vehicles:
                            speed = traci.vehicle.getSpeed(vid)
                            stats['total_vehicles'] += 1
                            total_speed_all += speed * 2.237  # Convert m/s to mph
                            
                            if speed <= STOPPED_THRESHOLD:
                                stats['stopped_vehicles'] += 1
                                stats['waiting_vehicles'] += 1
                            elif speed <= WAITING_THRESHOLD:
                                stats['waiting_vehicles'] += 1
                            else:
                                stats['moving_vehicles'] += 1
                                total_speed_moving += speed * 2.237
                    except traci.exceptions.TraCIException:
                        continue
            
            # Calculate averages
            if stats['moving_vehicles'] > 0:
                stats['avg_speed_moving'] = total_speed_moving / stats['moving_vehicles']
            if stats['total_vehicles'] > 0:
                stats['avg_speed_all'] = total_speed_all / stats['total_vehicles']
            
            return stats
        except Exception as e:
            print(f"Error getting traffic stats: {e}")
            return stats

    def _get_average_waiting_time(self):
        """Get the average waiting time for vehicles that are either stopped or moving slowly"""
        total_waiting_time = 0
        waiting_vehicles = 0
        WAITING_THRESHOLD = 0.3  # Lowered from 0.5 to 0.3 m/s to better filter actual waiting vs crawling
        
        try:
            for junction_name, edges in self.monitored_edges.items():
                for edge in edges:
                    try:
                        vehicles = traci.edge.getLastStepVehicleIDs(edge)
                        for veh_id in vehicles:
                            speed = traci.vehicle.getSpeed(veh_id)
                            if speed <= WAITING_THRESHOLD:  # Count both stopped and slow-moving vehicles
                                waiting_time = traci.vehicle.getWaitingTime(veh_id)
                                if waiting_time > 0:  # Only count if they've been waiting
                                    total_waiting_time += waiting_time
                                    waiting_vehicles += 1
                    except traci.exceptions.TraCIException:
                        continue
            
            return total_waiting_time / max(waiting_vehicles, 1)
        except Exception as e:
            print(f"Error getting average waiting time: {e}")
            return 0

    def _get_queue_lengths(self):
        """Get detailed queue lengths with critical lane tracking"""
        queue_lengths = {}
        junction_lane_waiting = {}
        total_queue = 0
        total_lanes = 0
        critical_lanes = 0
        
        try:
            for junction_name, edges in self.monitored_edges.items():
                junction_lane_waiting[junction_name] = {}
                junction_queue = 0
                
                for edge in edges:
                    try:
                        waiting_count = traci.edge.getLastStepHaltingNumber(edge)
                        queue_lengths[edge] = waiting_count
                        junction_lane_waiting[junction_name][edge] = waiting_count
                        
                        # Track critical lanes
                        if waiting_count > performance_targets['lane_thresholds']['critical']:
                            critical_lanes += 1
                        
                        junction_queue += waiting_count
                        total_lanes += 1
                    except traci.exceptions.TraCIException:
                        continue
                
                total_queue += junction_queue
                
                # Log per-junction metrics
                self.writer.add_scalar(f'Junction_{junction_name}/Queue_Length', 
                                     junction_queue, self.steps)
            
            # Calculate average queue per lane
            avg_queue_per_lane = total_queue / max(total_lanes, 1)
            critical_lane_ratio = critical_lanes / max(total_lanes, 1)
            
            # Log aggregate metrics
            self.writer.add_scalar('Metrics/Average_Queue_Per_Lane', 
                                 avg_queue_per_lane, self.steps)
            self.writer.add_scalar('Metrics/Critical_Lanes_Ratio', 
                                 critical_lane_ratio, self.steps)
            
            return queue_lengths, avg_queue_per_lane, junction_lane_waiting, critical_lane_ratio
        except Exception as e:
            print(f"Error getting queue lengths: {e}")
            return {}, 0, {}, 0

    def _update_metrics(self):
        """Update traffic metrics for the current simulation step"""
        try:
            # Update waiting times
            current_waiting_time = self._get_average_waiting_time()
            
            # Get detailed traffic statistics
            traffic_stats = self._get_traffic_stats()
            
            # Update speeds (now only for moving vehicles)
            current_speed = self._measure_traffic_speed()
            
            # Update queue lengths
            queue_lengths, avg_queue_per_lane, junction_lane_waiting, critical_lane_ratio = self._get_queue_lengths()
            
            # Update travel times
            self._update_travel_times()
            current_travel_time = self._get_average_travel_time()
            
            # Store all metrics
            self.episode_metrics['waiting_times'].append(current_waiting_time)
            self.episode_metrics['speeds'].append(current_speed)
            self.episode_metrics['queue_lengths'].append(avg_queue_per_lane)
            self.episode_metrics['travel_times'].append(current_travel_time)
            
            # Store additional statistics
            if 'traffic_stats' not in self.episode_metrics:
                self.episode_metrics['traffic_stats'] = []
            self.episode_metrics['traffic_stats'].append(traffic_stats)
            
            # Only print metrics at the end of each episode
            if self.steps >= self.max_steps:
                print(f"\nEpisode {self.episode + 1} Summary:")
                print(f"{'='*40}")
                print(f"Traffic Flow:")
                print(f"  • Total Vehicles: {traffic_stats['total_vehicles']}")
                print(f"  • Moving Vehicles: {traffic_stats['moving_vehicles']} ({traffic_stats['moving_vehicles']/max(traffic_stats['total_vehicles'],1)*100:.1f}%)")
                print(f"  • Stopped Vehicles: {traffic_stats['stopped_vehicles']} ({traffic_stats['stopped_vehicles']/max(traffic_stats['total_vehicles'],1)*100:.1f}%)")
                print(f"  • Waiting Vehicles: {traffic_stats['waiting_vehicles']} ({traffic_stats['waiting_vehicles']/max(traffic_stats['total_vehicles'],1)*100:.1f}%)")
                if self.episode_metrics['waiting_times']:
                    print(f"  • Average Waiting Time: {self.episode_metrics['waiting_times'][-1]:.2f}s (for waiting vehicles)")
                else:
                    print("Waiting Time: N/A")
                print(f"  • Average Queue per Lane: {avg_queue_per_lane:.2f} vehicles")
                print(f"Performance:")
                print(f"  • Average Travel Time: {current_travel_time:.2f}s")
                print(f"{'='*40}")
                
        except traci.exceptions.FatalTraCIError:
            self.done = True
        except Exception as e:
            print(f"Error updating metrics: {e}")
            self.done = True

    def step(self, action):
        """Execute one step in the environment. Now expects a dict of junction->dict with 'phase' and 'duration'."""
        global shutdown_requested
        
        # Check if shutdown was requested
        if shutdown_requested:
            self.done = True
            return self._get_state(), 0, True
            
        if self.done:
            return self._get_state(), 0, True
        try:
            train_present = self._check_train_presence()
            junction_actions = action
            current_time = self.steps * self.measurement_interval
            for junction_name, action_dict in junction_actions.items():
                if junction_name in self.controllable_junctions:
                    try:
                        junction_id = self.junctions[junction_name]
                        phase_index = action_dict['phase']
                        duration = action_dict['duration']
                        if junction_name == "rail_crossing" and train_present:
                            traci.trafficlight.setPhase(junction_id, 2)
                            self.current_phases[junction_name] = 2
                            self.last_phase_changes[junction_name] = current_time
                        elif current_time - self.last_phase_changes[junction_name] >= self.min_green_time:
                            if phase_index != self.current_phases[junction_name]:
                                traci.trafficlight.setPhase(junction_id, phase_index)
                                traci.trafficlight.setPhaseDuration(junction_id, duration)
                                self.current_phases[junction_name] = phase_index
                                self.last_phase_changes[junction_name] = current_time
                    except traci.exceptions.TraCIException:
                        continue
            traci.simulationStep()
            self.steps += 1
            if self.steps % self.measurement_interval == 0:
                self._update_metrics()
            next_state = self._get_state()
            reward = self._get_reward()
            done = self.steps >= self.max_steps
            
            # If episode is ending, ensure we have at least some metrics
            if done and not self.episode_metrics['speeds']:
                self._update_metrics()  # Force one final metrics update
                
            return next_state, reward, done
        except traci.exceptions.TraCIException:
            self.done = True
            return self._get_state(), 0, True
        except Exception as e:
            print(f"Error in step: {e}")
            self.done = True
            return self._get_state(), 0, True

    def _get_reward(self):
        try:
            if not self.episode_metrics['waiting_times'] or \
               not self.episode_metrics['speeds'] or \
               not self.episode_metrics['queue_lengths'] or \
               not self.episode_metrics['traffic_stats'] or \
               not self.episode_metrics['travel_times']:
                return 0.0

            current_waiting_time = self.episode_metrics['waiting_times'][-1]
            current_speed = self.episode_metrics['speeds'][-1]
            avg_queue = self.episode_metrics['queue_lengths'][-1]
            traffic_stats = self.episode_metrics['traffic_stats'][-1]
            stopped_vehicles = traffic_stats['stopped_vehicles']
            current_travel_time = self.episode_metrics['travel_times'][-1]

            # Default penalty for very short or long phases
            phase_penalty = 0.0
            for junction_name in self.controllable_junctions:
                try:
                    duration = traci.trafficlight.getPhaseDuration(self.junctions[junction_name])
                    if duration < 15:
                        phase_penalty -= 2.0
                    elif duration > 50:
                        phase_penalty -= 1.0
                except traci.exceptions.TraCIException:
                    continue

            # Define reward function variants (RF0–RF4)
            reward_modes = {
                'RF0': {'wait': -0.6, 'queue': -0.6, 'stopped': -0.3, 'speed': 1.2, 'travel': -0.3},  # Baseline
                'RF1': {'wait': -0.5, 'queue': -0.4, 'stopped': -0.4, 'speed': 1.0, 'travel': -0.3},  # Safety-focus
                'RF2': {'wait': -0.3, 'queue': -0.7, 'stopped': -0.2, 'speed': 1.3, 'travel': -0.2},  # Queue-focus
                'RF3': {'wait': -0.7, 'queue': -0.5, 'stopped': -0.2, 'speed': 1.1, 'travel': -0.1},  # Delay-minimization
                'RF4': {'wait': -0.4, 'queue': -0.3, 'stopped': -0.3, 'speed': 1.4, 'travel': -0.4},  # Speed/flow-focus
            }

            # Select the current mode weights
            mode = self.reward_mode
            if mode not in reward_modes:
                return 0.0
            weights = reward_modes[mode]

            # Compute weighted reward
            waiting_reward = weights['wait'] * current_waiting_time
            queue_reward = weights['queue'] * avg_queue
            stopped_reward = weights['stopped'] * stopped_vehicles
            speed_reward = weights['speed'] * current_speed
            travel_reward = weights['travel'] * current_travel_time

            reward = waiting_reward + queue_reward + stopped_reward + speed_reward + travel_reward + phase_penalty

            # Normalize by vehicle count and clip
            reward /= max(self.total_vehicles_passed, 1)
            reward = float(np.clip(reward, -100, 100))

            # Baseline adjustment
            self.running_reward_baseline = 0.99 * self.running_reward_baseline + 0.01 * reward
            reward -= self.running_reward_baseline

            return reward

        except Exception as e:
            print(f"[Reward Error] {e}")
            return 0.0

    def _initialize_metrics(self):
        """Initialize metrics tracking for the environment"""
        # Initialize episode metrics
        self.episode_metrics = {
            'rewards': [],
            'waiting_times': [],
            'speeds': [],
            'queue_lengths': [],
            'travel_times': [],
            'waiting_vehicles_count': []  # New metric to track number of waiting vehicles
        }
        
        # Initialize speed history
        self.speed_history = []
        
        # Initialize train-related metrics
        self.train_safety_score = 0.0
        self.train_approaching = False
        self.train_distance = float('inf')
        self.train_speed = 0.0
        
        # Initialize vehicle tracking
        self.vehicle_tracking = {}
        self.waiting_times = {}
        self.travel_times = {}
        self.queue_lengths = {}
        self.vehicle_travel_times = {}
        self.total_vehicles_passed = 0
        self.total_travel_time = 0
        
        # Initialize step counter
        self.steps = 0
        self.episode = 0
        self.done = False
        
        # Initialize state and action sizes
        self.state_size = 49
        self.action_size = 4

    def cleanup(self):
        """Clean up SUMO connection and resources"""
        try:
            if hasattr(self, 'connection_initialized') and self.connection_initialized and not self.connection_closed:
                # Check if TraCI connection is still valid
                try:
                    # Try a simple TraCI call to check connection
                    traci.simulation.getTime()
                    # If successful, close properly
                    traci.close()
                except (traci.exceptions.FatalTraCIError, ConnectionError, OSError):
                    # Connection already broken, just mark as closed
                    pass
                except Exception:
                    # Any other error, try to close anyway
                    try:
                        traci.close()
                    except:
                        pass
                
                self.connection_closed = True
                self.connection_initialized = False
                print("SUMO connection closed successfully")
        except Exception as e:
            print(f"Warning: Error during SUMO cleanup: {e}")
            self.connection_closed = True
            self.connection_initialized = False

    def close(self):
        """Close the environment and save final plots"""
        try:
            # Clean up SUMO connection
            self.cleanup()
            
            # Close TensorBoard writer
            if hasattr(self, 'writer') and self.writer is not None:
                self.writer.close()
        except Exception as e:
            print(f"Error closing environment: {e}")

class DQNAgent:
    def __init__(self, state_size, junction_actions, device, td_error_agg='mean'):
        self.state_size = state_size
        self.junction_actions = junction_actions
        self.device = device
        
        # Network parameters
        self.hidden_size = 256
        self.learning_rate = 0.001
        
        # Initialize networks with correct state size
        self.policy_net = DQN(state_size=49, hidden_size=self.hidden_size, junction_actions=junction_actions).to(device)
        self.target_net = DQN(state_size=49, hidden_size=self.hidden_size, junction_actions=junction_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Optimizer with gradient clipping
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        
        # Experience replay with prioritized sampling
        self.memory = PrioritizedReplayBuffer(10000, alpha=0.6, beta=0.4, td_error_agg=td_error_agg)
        
        # Training parameters
        self.batch_size = 64
        self.gamma = 0.99
        self.target_update = 10
        self.steps_done = 0
        
        # Modified exploration strategy with extended decay
        self.eps_start = 1.0
        self.eps_end = 0.05
        self.eps_decay = 12000  # Extended from 8000 to 12000 for slower exploration decay
        
        # Prioritized replay parameters
        self.priority_alpha = 0.6
        self.priority_beta = 0.4
        self.priority_beta_increment = 0.001
        
        # Gradient clipping
        self.max_grad_norm = 1.0
        
        # Huber loss for stability
        self.criterion = nn.SmoothL1Loss()
        
    def select_action(self, state):
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1
        
        if random.random() > eps_threshold:
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                outputs = self.policy_net(state)
                return {
                    j: {
                        'phase': outputs[j]['phase_logits'].argmax(1).item(),
                        'duration': int(torch.clamp(outputs[j]['duration'], 10, 60).item())
                    }
                    for j in self.junction_actions
                }
        else:
            return {
                j: {
                    'phase': random.randint(0, n - 1),
                    'duration': random.randint(10, 60)
                }
                for j, n in self.junction_actions.items()
            }

    def learn(self):
        if len(self.memory) < self.batch_size:
            return
            
        # Sample batch from memory
        transitions, indices, weights = self.memory.sample(self.batch_size)
        
        # Convert batch to tensors
        state_batch = torch.cat([torch.FloatTensor(t[0]).view(1, -1) for t in transitions]).to(self.device)
        # actions are now dicts per junction
        action_dicts = [t[1] for t in transitions]
        reward_batch = torch.FloatTensor([t[2] for t in transitions]).to(self.device)
        next_state_batch = torch.cat([torch.FloatTensor(t[3]).view(1, -1) for t in transitions]).to(self.device)
        done_batch = torch.FloatTensor([t[4] for t in transitions]).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)
        
        # For each junction, build a batch of actions
        current_q_values = {}
        for junction in self.junction_actions.keys():
            # Use only the 'phase_logits' tensor for Q-value selection
            junction_q_values = self.policy_net(state_batch)[junction]['phase_logits']
            junction_phase_actions = [a[junction]['phase'] for a in action_dicts]
            junction_action_batch = torch.LongTensor(junction_phase_actions).to(self.device)
            current_q_values[junction] = junction_q_values.gather(1, junction_action_batch.unsqueeze(1))
        # Get next Q values for each junction
        with torch.no_grad():
            next_policy_q = self.policy_net(next_state_batch)
            next_actions = {}
            for junction in self.junction_actions:
                next_actions[junction] = next_policy_q[junction]['phase_logits'].argmax(1, keepdim=True)
            next_target_q = self.target_net(next_state_batch)
            next_q_values = {}
            for junction in self.junction_actions:
                next_q_values[junction] = next_target_q[junction]['phase_logits'].gather(1, next_actions[junction])
        
        # Calculate target Q values
        target_q_values = {}
        for junction in self.junction_actions:
            target_q_values[junction] = reward_batch.unsqueeze(1) + \
                (1 - done_batch.unsqueeze(1)) * self.gamma * next_q_values[junction]
        
        # Calculate loss for each junction
        losses = []
        q_value_means = {}
        for junction in self.junction_actions.keys():
            loss = F.smooth_l1_loss(current_q_values[junction], target_q_values[junction])
            losses.append(loss)
            # Log Q-value mean for TensorBoard
            q_value_means[junction] = current_q_values[junction].mean().item()
        # Average the losses
        loss = torch.stack(losses).mean()
        # TensorBoard logging
        if hasattr(self, 'writer') and self.writer is not None:
            self.writer.add_scalar('Loss', loss.item(), self.steps_done)
            for junction, q_mean in q_value_means.items():
                self.writer.add_scalar(f'QValue/{junction}', q_mean, self.steps_done)
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        # Update priorities in PrioritizedReplayBuffer using TD-error
        td_errors = []
        for junction in self.junction_actions:
            td = target_q_values[junction] - current_q_values[junction]
            td_errors.append(td.detach().squeeze().cpu().numpy())
        if hasattr(self.memory, 'td_error_agg') and self.memory.td_error_agg == 'max':
            td_errors_flat = np.max(np.abs(td_errors), axis=0)
        else:
            td_errors_flat = np.mean(np.abs(td_errors), axis=0)
        self.memory.update_priorities(indices, td_errors_flat)
        
        # Update target network if needed
        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, filename):
        torch.save(self.policy_net.state_dict(), filename)

    def load(self, filename):
        """Load the model state from a file"""
        try:
            # Load the state dict from the file
            state_dict = torch.load(filename, map_location=self.device)
            self.policy_net.load_state_dict(state_dict)
            self.target_net.load_state_dict(state_dict)
            print(f"Successfully loaded model from {filename}")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Initializing new model state")
            # Initialize new state dict if loading fails
            self.policy_net = DQN(state_size=49, hidden_size=self.hidden_size, junction_actions=self.junction_actions).to(self.device)
            self.target_net = DQN(state_size=49, hidden_size=self.hidden_size, junction_actions=self.junction_actions).to(self.device)
            self.target_net.load_state_dict(self.policy_net.state_dict())

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4, td_error_agg='mean'):
        self.capacity = capacity
        self.alpha = alpha  # Priority exponent
        self.beta = beta    # Importance sampling exponent
        self.beta_increment = 0.001
        self.memory = []
        self.priorities = np.zeros(capacity)
        self.position = 0
        self.td_error_agg = td_error_agg  # 'mean' or 'max'
    
    def push(self, *args):
        max_priority = np.max(self.priorities) if self.memory else 1.0
        
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = args
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        if len(self.memory) == 0:
            return None, None, None
        
        probs = self.priorities[:len(self.memory)] ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.memory), batch_size, p=probs)
        samples = [self.memory[idx] for idx in indices]
        
        # Compute importance weights
        total = len(self.memory)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        
        return samples, indices, weights
    
    def update_priorities(self, indices, td_errors):
        for idx, td_error in zip(indices, td_errors):
            self.priorities[idx] = td_error + 1e-6
    
    def __len__(self):
        return len(self.memory)

def print_progress(step, total_steps, episode):
    """Simple progress indicator"""
    progress = (step / total_steps) * 100
    print(f"\rEpisode {episode} - Progress: {progress:.1f}% ({step}/{total_steps})", end='')

def save_episode_plots(episode, plots_dir, rewards_history, waiting_times_history, 
                      avg_speeds_history, queue_lengths_history, travel_times_history, 
                      epsilon_history, stopped_vehicles_history):
    """Save plots for the current episode state"""
    # Create episode-specific directory
    episode_dir = os.path.join(plots_dir, f'episode_{episode+1}')
    os.makedirs(episode_dir, exist_ok=True)
    
    # Plot metrics
    plt.figure(figsize=(12, 6))
    episodes = np.arange(1, len(avg_speeds_history) + 1)
    smoothed_speeds = ema(avg_speeds_history)
    smoothed_waiting = ema(waiting_times_history)
    smoothed_queue = ema(queue_lengths_history)
    smoothed_travel = ema(travel_times_history)
    
    plt.plot(episodes, smoothed_speeds, label='Smoothed Avg Speed')
    plt.plot(episodes, smoothed_waiting, label='Smoothed Waiting Time')
    plt.plot(episodes, smoothed_queue, label='Smoothed Queue Length')
    plt.plot(episodes, smoothed_travel, label='Smoothed Travel Time')
    plt.xlabel('Episode')
    plt.ylabel('Value')
    plt.title(f'Smoothed Episode Averages (Episode {episode+1})')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(episode_dir, 'metrics.png'))
    plt.close()
    
    # Plot rewards
    plt.figure(figsize=(12, 6))
    plt.plot(rewards_history, label='Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title(f'Training Rewards (Episode {episode+1})')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(episode_dir, 'rewards.png'))
    plt.close()
    
    # Plot epsilon decay
    plt.figure(figsize=(12, 4))
    plt.plot(range(1, len(epsilon_history) + 1), epsilon_history, label='Epsilon')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    plt.title(f'Epsilon Decay (Episode {episode+1})')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(episode_dir, 'epsilon.png'))
    plt.close()
    
    # Multi-metric overview plot for this episode (3x2 grid, no duplicates)
    if not (len(rewards_history) == len(waiting_times_history) == len(avg_speeds_history) == 
            len(queue_lengths_history) == len(travel_times_history) == len(epsilon_history) == len(stopped_vehicles_history)) or len(stopped_vehicles_history) == 0:
        print("Skipping multi-metric plot: histories are not the same length or stopped_vehicles_history is empty.")
        return
    plot_training_metrics(
        rewards_history, waiting_times_history, avg_speeds_history,
        queue_lengths_history, travel_times_history, epsilon_history, stopped_vehicles_history, episode_dir
    )
    
    print(f"\nSaved episode {episode+1} plots to: {episode_dir}")

def train(episodes=600, resume_from=None, min_episodes=100, patience=50, target_reward=15, save_csv=True, reward_mode='baseline'):
    """Train the DQN agent with early stopping based on performance"""
    print(f"\n=== Starting Training for {reward_mode.upper()} variant ===")
    
    # Create variant-specific output directory
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output', f'output_{reward_mode}')
    os.makedirs(output_dir, exist_ok=True)
    
    # Create plots directory for this variant
    plots_dir = os.path.join(output_dir, 'plots')
    print(f"\nCreating plots directory at: {plots_dir}")
    os.makedirs(plots_dir, exist_ok=True)
    print(f"Plots directory created/verified at: {plots_dir}")
    
    print(f"Maximum episodes: {episodes}")
    print(f"Minimum episodes: {min_episodes}")
    print(f"Patience: {patience}")
    print(f"Target reward: {target_reward}")
    print(f"Using device: {device}")
    print(f"Reward mode: {reward_mode}")
    
    # Initialize environment with shorter episodes
    env = TrafficEnvironment(sumocfg_file, max_steps=3600, reward_mode=reward_mode)  # 60 minutes per episode
    
    # Initialize agent with optimized learning parameters
    agent = DQNAgent(
        state_size=49,
        junction_actions=env.junction_actions,
        device=device,
        td_error_agg='max'
    )
    
    # Lists to store metrics
    rewards_history = []
    waiting_times_history = []
    avg_speeds_history = []
    queue_lengths_history = []
    travel_times_history = []
    stopped_vehicles_history = []
    epsilon_history = []
    
    # Training statistics
    running_reward = 0
    running_waiting_time = 0
    running_speed = 0
    running_travel_time = 0
    running_queue_length = 0
    
    # Early stopping variables
    best_reward = float('-inf')
    no_improvement_count = 0
    should_stop = False
    
    # Performance tracking
    performance_achieved = False
    performance_stable_count = 0
    
    # Resume logic
    start_episode = 0
    if resume_from is not None:
        print(f"\nLoading checkpoint from {resume_from} ...")
        if not os.path.exists(resume_from):
            print(f"Checkpoint file not found: {resume_from}")
            return None, [], [], [], [], []
        
        checkpoint = torch.load(resume_from, map_location=device, weights_only=False)
        agent.policy_net.load_state_dict(checkpoint['model_state_dict'])
        agent.target_net.load_state_dict(checkpoint['model_state_dict'])
        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        agent.steps_done = checkpoint.get('steps_done', 0)
        start_episode = checkpoint.get('episode', 0) + 1  # Resume from next episode
        # Restore histories if present
        rewards_history = checkpoint.get('rewards_history', [])
        waiting_times_history = checkpoint.get('waiting_times_history', [])
        avg_speeds_history = checkpoint.get('avg_speeds_history', [])
        queue_lengths_history = checkpoint.get('queue_lengths_history', [])
        travel_times_history = checkpoint.get('travel_times_history', [])
        # If stopped_vehicles_history or epsilon_history are not present, fill with zeros
        stopped_vehicles_history = checkpoint.get('stopped_vehicles_history', [0]*len(rewards_history))
        epsilon_history = checkpoint.get('epsilon_history', [0]*len(rewards_history))
        print(f"Resuming from episode {start_episode} (checkpoint episode {checkpoint.get('episode', 0)})")
    
    try:
        for episode in range(start_episode, episodes):
            # Check if shutdown was requested
            if shutdown_requested:
                print("\nShutdown requested. Stopping training gracefully...")
                break
                
            print(f"\nEpisode {episode + 1}/{episodes}")
            state = env.reset()
            episode_reward = 0
            step_count = 0
            
            while True:
                # Check if shutdown was requested
                if shutdown_requested:
                    print("\nShutdown requested during episode. Stopping gracefully...")
                    break
                    
                actions = agent.select_action(state)
                next_state, reward, done = env.step(actions)
                agent.memory.push(state, actions, float(reward), next_state, float(done))
                state = next_state
                episode_reward += reward
                step_count += 1
                agent.learn()
                
                # Update progress
                print_progress(step_count, env.max_steps, episode + 1)
                
                if done:
                    print()  # New line after progress
                    break
            
            # Clean up SUMO connection after each episode
            env.cleanup()
            
            # If shutdown was requested, break out of episode loop
            if shutdown_requested:
                break
            
            # Print episode summary
            print(f"\n{'='*50}")
            print(f"Episode {episode + 1} Summary:")
            print(f"{'='*50}")
            print(f"Reward: {episode_reward:.2f}")
            if env.episode_metrics['waiting_times']:
                print(f"Waiting Time: {env.episode_metrics['waiting_times'][-1]:.2f}s")
            else:
                print("Waiting Time: N/A")
            if env.episode_metrics['speeds']:
                print(f"Speed: {env.episode_metrics['speeds'][-1]:.2f} mph")
            else:
                print("Speed: N/A")
            if env.episode_metrics['queue_lengths']:
                print(f"Queue Length: {env.episode_metrics['queue_lengths'][-1]:.2f}")
            else:
                print("Queue Length: N/A")
            if env.episode_metrics['travel_times']:
                print(f"Travel Time: {env.episode_metrics['travel_times'][-1]:.2f}s")
            else:
                print("Travel Time: N/A")
            print(f"Epsilon: {agent.eps_end + (agent.eps_start - agent.eps_end) * math.exp(-1. * agent.steps_done / agent.eps_decay):.3f}")
            print(f"Steps: {step_count}")
            print(f"Running Average Reward: {running_reward:.2f}")
            print(f"{'='*50}\n")
            
            # Save checkpoint every 10 episodes
            if (episode + 1) % 10 == 0:
                checkpoint = {
                    'model_state_dict': agent.policy_net.state_dict(),
                    'optimizer_state_dict': agent.optimizer.state_dict(),
                    'steps_done': agent.steps_done,
                    'episode': episode,
                    'rewards_history': rewards_history,
                    'waiting_times_history': waiting_times_history,
                    'avg_speeds_history': avg_speeds_history,
                    'queue_lengths_history': queue_lengths_history,
                    'travel_times_history': travel_times_history,
                    'stopped_vehicles_history': stopped_vehicles_history,
                    'epsilon_history': epsilon_history
                }
                checkpoint_path = os.path.join(output_dir, f'checkpoint_episode_{episode+1}.pth')
                torch.save(checkpoint, checkpoint_path)
                print(f"\nCheckpoint saved at episode {episode+1}: {checkpoint_path}")
            
            # Update running averages
            running_reward = 0.9 * running_reward + 0.1 * episode_reward
            if env.episode_metrics['waiting_times']:
                running_waiting_time = 0.9 * running_waiting_time + 0.1 * env.episode_metrics['waiting_times'][-1]
            if env.episode_metrics['speeds']:
                running_speed = 0.9 * running_speed + 0.1 * env.episode_metrics['speeds'][-1]
            if env.episode_metrics['travel_times']:
                running_travel_time = 0.9 * running_travel_time + 0.1 * env.episode_metrics['travel_times'][-1]
            if env.episode_metrics['queue_lengths']:
                running_queue_length = 0.9 * running_queue_length + 0.1 * env.episode_metrics['queue_lengths'][-1]
            
            # Calculate episode averages
            avg_speed = np.mean(env.episode_metrics['speeds']) if env.episode_metrics['speeds'] else 0.0
            avg_waiting_time = np.mean(env.episode_metrics['waiting_times']) if env.episode_metrics['waiting_times'] else 0.0
            avg_queue_length = np.mean(env.episode_metrics['queue_lengths']) if env.episode_metrics['queue_lengths'] else 0.0
            avg_travel_time = np.mean(env.episode_metrics['travel_times']) if env.episode_metrics['travel_times'] else 0.0
            avg_stopped_vehicles = 0.0
            if 'traffic_stats' in env.episode_metrics and env.episode_metrics['traffic_stats']:
                avg_stopped_vehicles = np.mean([s.get('stopped_vehicles', 0) for s in env.episode_metrics['traffic_stats']])
            
            # Store metrics (single append)
            rewards_history.append(episode_reward)
            waiting_times_history.append(avg_waiting_time)
            avg_speeds_history.append(avg_speed)
            queue_lengths_history.append(avg_queue_length)
            travel_times_history.append(avg_travel_time)
            stopped_vehicles_history.append(avg_stopped_vehicles)
            
            # After each episode, calculate and store epsilon
            eps_threshold = agent.eps_end + (agent.eps_start - agent.eps_end) * math.exp(-1. * agent.steps_done / agent.eps_decay)
            epsilon_history.append(eps_threshold)
            
            # Save plots after all metrics are updated
            save_episode_plots(
                episode, plots_dir, rewards_history, waiting_times_history,
                avg_speeds_history, queue_lengths_history, travel_times_history,
                epsilon_history, stopped_vehicles_history
            )
            
            print(f"Episode {episode+1} averages: Speed={avg_speed:.2f}, Waiting Time={avg_waiting_time:.2f}, Queue Length={avg_queue_length:.2f}, Travel Time={avg_travel_time:.2f}, Stopped Vehicles={avg_stopped_vehicles:.2f}")
            
            # Check if performance targets are achieved
            if episode >= min_episodes:
                waiting_time_ok = performance_targets['waiting_time']['excellent'] <= running_waiting_time <= performance_targets['waiting_time']['critical']
                speed_ok = running_speed >= performance_targets['speed']['excellent'] * 0.95
                queue_ok = running_queue_length <= performance_targets['lane_thresholds']['critical']
                
                if waiting_time_ok and speed_ok and queue_ok:
                    performance_stable_count += 1
                    if performance_stable_count >= 10:
                        performance_achieved = True
                        print("\nPerformance targets achieved and stable!")
                        should_stop = True
                else:
                    performance_stable_count = 0
            
            # Save best model
            if running_reward > best_reward:
                best_reward = running_reward
                checkpoint = {
                    'model_state_dict': agent.policy_net.state_dict(),
                    'optimizer_state_dict': agent.optimizer.state_dict(),
                    'steps_done': agent.steps_done,
                    'episode': episode,
                    'rewards_history': rewards_history,
                    'waiting_times_history': waiting_times_history,
                    'avg_speeds_history': avg_speeds_history,
                    'queue_lengths_history': queue_lengths_history,
                    'travel_times_history': travel_times_history,
                    'stopped_vehicles_history': stopped_vehicles_history,
                    'epsilon_history': epsilon_history
                }
                best_model_path = os.path.join(output_dir, 'best_model.pth')
                torch.save(checkpoint, best_model_path)
                print(f"\nNew best model saved! Reward: {best_reward:.2f}")
                print(f"Model saved to: {best_model_path}")
            
            if should_stop:
                print("\nTraining stopped - Performance targets achieved!")
                break
    
            # Monitor reward standard deviation
            window = 20  # or any window size you prefer
            if len(rewards_history) >= window:
                recent_rewards = rewards_history[-window:]
                reward_std = np.std(recent_rewards)
                reward_mean = np.mean(recent_rewards)
                print(f"Recent Reward Mean: {reward_mean:.2f}, Std: {reward_std:.2f}")
                # Use reward_std < threshold as a convergence criterion
            
            convergence_window = 20
            convergence_std_threshold = 0.5  # or another small value

            if len(rewards_history) >= convergence_window:
                recent_rewards = rewards_history[-convergence_window:]
                reward_std = np.std(recent_rewards)
                if reward_std < convergence_std_threshold:
                    print("Converged: reward std below threshold.")
                    break
            
            # After storing metrics and before early stopping checks:
            if len(rewards_history) >= 20:
                std = np.std(rewards_history[-20:])
                print(f"Reward Std Dev (last 20): {std:.2f}")
            
    finally:
        env.close()
    
    if shutdown_requested:
        print("\n=== Training Interrupted ===")
        print("Training was stopped due to user interruption.")
    else:
        print("\n=== Training Completed ===")
    print(f"Final Running Average Reward: {running_reward:.2f}")
    print(f"Best Reward Achieved: {best_reward:.2f}")
    print(f"Total Episodes: {episode + 1}")
    
    # Save final plots in the main plots directory
    print("\nGenerating final plots...")
    save_episode_plots(
        episode, plots_dir, rewards_history, waiting_times_history,
        avg_speeds_history, queue_lengths_history, travel_times_history,
        epsilon_history, stopped_vehicles_history
    )
    
    # Save all-metrics overview plot
    fig, axs = plt.subplots(3, 2, figsize=(16, 10))
    # Create proper episodes array for x-axis
    episodes = np.arange(1, len(rewards_history) + 1)  # 1-based episode numbers
    
    axs[0, 0].plot(episodes, rewards_history, color='#1f77b4', marker='o', markevery=10, label='Reward')
    axs[0, 0].set_title('Average Reward per Episode')
    axs[0, 0].set_xlabel('Episode')
    axs[0, 0].set_ylabel('Average Reward')
    axs[0, 0].grid(True, linestyle='--', alpha=0.7)

    axs[0, 1].plot(episodes, queue_lengths_history, color='#ff7f0e', marker='s', markevery=10, label='Queue Length')
    axs[0, 1].set_title('Average Queue Length per Episode')
    axs[0, 1].set_xlabel('Episode')
    axs[0, 1].set_ylabel('Average Queue Length')
    axs[0, 1].grid(True, linestyle='--', alpha=0.7)

    axs[1, 0].plot(episodes, stopped_vehicles_history, color='#2ca02c', marker='^', markevery=10, label='Stopped Vehicles')
    axs[1, 0].set_title('Avg Stopped Vehicles per Episode')
    axs[1, 0].set_xlabel('Episode')
    axs[1, 0].set_ylabel('Stopped Vehicles')
    axs[1, 0].grid(True, linestyle='--', alpha=0.7)
    axs[1, 0].legend()

    axs[1, 1].plot(episodes, waiting_times_history, color='#d62728', marker='D', markevery=10, label='Waiting Time')
    axs[1, 1].set_title('Average Waiting Time per Episode')
    axs[1, 1].set_xlabel('Episode')
    axs[1, 1].set_ylabel('Average Waiting Time (s)')
    axs[1, 1].grid(True, linestyle='--', alpha=0.7)

    axs[2, 0].plot(episodes, avg_speeds_history, color='#9467bd', marker='P', markevery=10, label='Speed')
    axs[2, 0].set_title('Average Speed per Episode')
    axs[2, 0].set_xlabel('Episode')
    axs[2, 0].set_ylabel('Average Speed (mph)')
    axs[2, 0].grid(True, linestyle='--', alpha=0.7)

    axs[2, 1].plot(episodes, travel_times_history, color='#8c564b', marker='X', markevery=10, label='Travel Time')
    axs[2, 1].set_title('Average Travel Time per Episode')
    axs[2, 1].set_xlabel('Episode')
    axs[2, 1].set_ylabel('Average Travel Time (s)')
    axs[2, 1].grid(True, linestyle='--', alpha=0.7)
    axs[2, 1].legend()

    axs[2, 1].legend()

    plt.tight_layout()
    save_path = os.path.join(plots_dir, 'all_metrics_overview.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Saved all-metrics overview plot to: {save_path}")

    # Optional: dump metrics as CSV
    if save_csv:
        df = pd.DataFrame({
            'reward': rewards_history,
            'waiting_time': waiting_times_history,
            'avg_speed': avg_speeds_history,
            'queue_length': queue_lengths_history,
            'travel_time': travel_times_history,
            'stopped_vehicles': stopped_vehicles_history,
            'epsilon': epsilon_history
        })
        csv_path = os.path.join(output_dir, 'training_metrics.csv')
        df.to_csv(csv_path, index=False)
        print(f"Saved training metrics to {csv_path}")

    return agent, rewards_history, waiting_times_history, avg_speeds_history, epsilon_history, stopped_vehicles_history

def ema(values, alpha=0.1):
    ema_values = []
    for i, v in enumerate(values):
        if i == 0:
            ema_values.append(v)
        else:
            ema_values.append(alpha * v + (1 - alpha) * ema_values[-1])
    return ema_values

def plot_training_metrics(
    rewards_history, waiting_times_history, avg_speeds_history,
    queue_lengths_history, travel_times_history, epsilon_history, stopped_vehicles_history, save_dir
):
    episodes = np.arange(1, len(rewards_history) + 1)  # 1-based episode numbers
    plt.style.use('seaborn-v0_8-colorblind')  # Modern, colorblind-friendly style
    colors = [
        '#1f77b4',  # blue
        '#ff7f0e',  # orange
        '#2ca02c',  # green
        '#d62728',  # red
        '#9467bd',  # purple
        '#8c564b',  # brown
        '#e377c2',  # pink (for stopped vehicles)
    ]
    markers = ['o', 's', '^', 'D', 'P', 'X', '*']
    fig, axs = plt.subplots(3, 2, figsize=(16, 10))
    fig.tight_layout(pad=4.0)

    axs[0, 0].plot(episodes, rewards_history, color=colors[0], marker=markers[0], markevery=10, label='Reward')
    axs[0, 0].set_title('Average Reward per Episode')
    axs[0, 0].set_xlabel('Episode')
    axs[0, 0].set_ylabel('Average Reward')
    axs[0, 0].grid(True, linestyle='--', alpha=0.7)

    axs[0, 1].plot(episodes, queue_lengths_history, color=colors[1], marker=markers[1], markevery=10, label='Queue Length')
    axs[0, 1].set_title('Average Queue Length per Episode')
    axs[0, 1].set_xlabel('Episode')
    axs[0, 1].set_ylabel('Average Queue Length')
    axs[0, 1].grid(True, linestyle='--', alpha=0.7)

    axs[1, 0].plot(episodes, stopped_vehicles_history, color=colors[6], marker=markers[6], markevery=10, label='Stopped Vehicles')
    axs[1, 0].set_title('Avg Stopped Vehicles per Episode')
    axs[1, 0].set_xlabel('Episode')
    axs[1, 0].set_ylabel('Stopped Vehicles')
    axs[1, 0].grid(True, linestyle='--', alpha=0.7)
    axs[1, 0].legend()

    axs[1, 1].plot(episodes, waiting_times_history, color=colors[2], marker=markers[2], markevery=10, label='Waiting Time')
    axs[1, 1].set_title('Average Waiting Time per Episode')
    axs[1, 1].set_xlabel('Episode')
    axs[1, 1].set_ylabel('Average Waiting Time (s)')
    axs[1, 1].grid(True, linestyle='--', alpha=0.7)

    axs[2, 0].plot(episodes, avg_speeds_history, color=colors[3], marker=markers[3], markevery=10, label='Speed')
    axs[2, 0].set_title('Average Speed per Episode')
    axs[2, 0].set_xlabel('Episode')
    axs[2, 0].set_ylabel('Average Speed (mph)')
    axs[2, 0].grid(True, linestyle='--', alpha=0.7)

    axs[2, 1].plot(episodes, travel_times_history, color=colors[4], marker=markers[4], markevery=10, label='Travel Time')
    axs[2, 1].set_title('Average Travel Time per Episode')
    axs[2, 1].set_xlabel('Episode')
    axs[2, 1].set_ylabel('Average Travel Time (s)')
    axs[2, 1].grid(True, linestyle='--', alpha=0.7)
    axs[2, 1].legend()

    # Add legends for each subplot
    for ax in axs.flat:
        ax.legend()

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'all_metrics_overview.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Saved all-metrics overview plot to: {save_path}")

# Set cuDNN benchmark if using CUDA
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    print("cuDNN benchmark enabled for optimal performance on CUDA.")

# Utility function to profile float32 vs bfloat16

def profile_dtype_performance(model, input_shape=(1, 49), device='cuda'):
    """Profile inference speed for float32 vs bfloat16 if supported."""
    import time
    if not torch.cuda.is_available():
        print("CUDA not available, skipping dtype profiling.")
        return
    x = torch.randn(input_shape).to(device)
    model = model.to(device)
    model.eval()
    # Warmup
    for _ in range(10):
        _ = model(x)
    # Profile float32
    start = time.time()
    for _ in range(100):
        _ = model(x)
    t32 = time.time() - start
    print(f"float32 inference time: {t32:.4f}s")
    # Profile bfloat16 if supported
    if torch.cuda.is_bf16_supported():
        x_bf16 = x.to(dtype=torch.bfloat16)
        model_bf16 = model.to(dtype=torch.bfloat16)
        for _ in range(10):
            _ = model_bf16(x_bf16)
        start = time.time()
        for _ in range(100):
            _ = model_bf16(x_bf16)
        tbf16 = time.time() - start
        print(f"bfloat16 inference time: {tbf16:.4f}s")
    else:
        print("bfloat16 not supported on this device.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train traffic control agent')
    parser.add_argument('--resume_from', type=str, help='Path to checkpoint file to resume from')
    parser.add_argument('--episodes', type=int, default=600, help='Number of episodes to train')
    parser.add_argument('--reward_mode', type=str, default='baseline', help='Reward function mode to use')
    args = parser.parse_args()
    
    print("\nStarting training with performance-focused parameters...")
    print("\nPerformance Targets:")
    print("Speed:")
    print(f"  • Target: {performance_targets['speed']['target']} mph")
    print(f"  • Excellent: ≥ {performance_targets['speed']['excellent']} mph")
    print(f"  • Good: ≥ {performance_targets['speed']['good']} mph")
    print(f"  • Poor: ≥ {performance_targets['speed']['poor']} mph")
    print(f"  • Critical: < {performance_targets['speed']['critical']} mph")
    print("\nWaiting Time:")
    print(f"  • Excellent: ≤ {performance_targets['waiting_time']['excellent']} seconds")
    print(f"  • Good: ≤ {performance_targets['waiting_time']['good']} seconds")
    print(f"  • Critical: ≤ {performance_targets['waiting_time']['critical']} seconds")
    print("\nLane Thresholds:")
    print(f"  • Excellent: ≤ {performance_targets['lane_thresholds']['excellent']} vehicles/lane")
    print(f"  • Good: ≤ {performance_targets['lane_thresholds']['good']} vehicles/lane")
    print(f"  • Critical: ≤ {performance_targets['lane_thresholds']['critical']} vehicles/lane")
    
    if args.resume_from:
        print(f"\nResuming training from checkpoint: {args.resume_from}")
    
    agent, rewards_history, waiting_times_history, avg_speeds_history, epsilon_history, stopped_vehicles_history = train(
        episodes=args.episodes,
        resume_from=args.resume_from,
        min_episodes=100,
        patience=50,
        target_reward=15,
        save_csv=True,
        reward_mode=args.reward_mode
    )
