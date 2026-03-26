import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import namedtuple, deque
import random
import warnings
import math
import time
import torch.nn.functional as F
import pandas as pd # type: ignore
# Robust device import
try:
    from traffic_components import device
except ImportError:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import argparse
import subprocess

warnings.filterwarnings("ignore")

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

# Check for SUMO_HOME environment variable
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
            # Define possible routes and their edges (keeping the same routes as in osm.rou.xml)
            routes = {
                "route1": ["19448704#00", "683562137#1", "683562137#2", "634020683"],
                "route2": ["39447439#0", "39447439#1", "683562128", "683562137#0", "683562137#1", "683562137#2", "634020683"],
                "route3": ["-19470790#0", "935204071#1", "-1347512024", "-1347512023#1", "-1347512023#0", "-1346082739"],
                "route4": ["245915228#1", "-245915228#2", "-19468172"],
                "route5": ["340808718#10.50", "683562137#2", "-1346082739", "-634020683"],
                # Add all other routes from osm.rou.xml
                "route6": ["634020673#0", "-340808718#0", "E0", "E0.79", "1346606197", "245915228#0", "245915228#1", "245915228#2", "245915228#3", "245915228#4", "-1346338965", "-1346082738#3", "-1346082738#2", "-1346082738#1", "E1", "E1.78", "-1347512024", "-1347512023#1", "-1347512023#0", "-1346082739", "-340808718#10", "-340808718#100", "-340808718#9", "-340808718#8", "-340808718#7", "-340808718#6", "-340808718#5", "-340808718#4", "-340808718#3", "-340808718#2", "-340808718#1"],
                # ... (add all other routes)
            }
            
            # Original flow rates from osm.rou.xml
            original_flows = {
                "f_0": 300, "f_2": 250, "f_3": 30, "f_4": 70, "f_5": 25, "f_6": 25,
                "f_7": 25, "f_8": 25, "f_9": 25, "f_10": 10, "f_11": 15, "f_12": 25,
                "f_13": 25, "f_14": 25, "f_15": 25, "f_16": 45, "f_17": 5, "f_18": 25,
                "f_19": 25, "f_20": 50, "f_21": 50, "f_22": 50, "f_23": 50, "f_24": 50,
                "f_25": 50, "f_26": 5, "f_27": 25, "f_28": 25, "f_29": 50, "f_30": 50,
                "f_31": 50, "f_32": 50, "f_33": 50, "f_34": 50, "f_35": 50, "f_36": 75,
                "f_37": 75, "f_38": 50, "f_39": 75, "f_40": 60, "f_41": 40
            }
            
            # Create route file content with fixed train schedule
            route_file_content = """<?xml version="1.0" encoding="UTF-8"?>
<routes>
    <!-- Vehicle Types -->
    <vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="5" minGap="2.5" maxSpeed="17.88" guiShape="passenger"/>
    <vType id="DEFAULT_RAILTYPE" length="2286.00" desiredMaxSpeed="8.94" vClass="rail"/>
"""
            
            # Collect all flows to sort them by departure time
            flows = []
            total_vehicles = 0  # Track total vehicles
            
            # Add car flows with randomized rates (±15% variation)
            for route_id, edges in routes.items():
                via = " ".join(edges[1:-1]) if len(edges) > 2 else ""
                original_rate = original_flows.get(f"f_{route_id[-1]}", 25)
                
                # Special handling for fixed number flows (like f_2)
                if route_id == "route2":  # This is f_2
                    flow_rate = 250  # Keep fixed
                else:
                    # Add more variation for other flows
                    variation = np.random.uniform(0.85, 1.15)  # ±15% variation
                    flow_rate = int(original_rate * variation)
                
                # Calculate vehicles for this flow (1 hour duration)
                vehicles_per_flow = int(flow_rate)  # vehsPerHour for 1 hour = total vehicles
                total_vehicles += vehicles_per_flow
                
                flow_content = f'    <flow id="f_{route_id}" begin="0.00" from="{edges[0]}" to="{edges[-1]}"'
                if via:
                    flow_content += f' via="{via}"'
                flow_content += f' end="3600.00" vehsPerHour="{flow_rate}" type="car"/>'
                flows.append((0.0, flow_content))
            
            # Add train flows (sorted by departure time)
            train_flows = [
                (1200.0, '    <flow id="f_train1" type="DEFAULT_RAILTYPE" begin="1200.00" from="832911536#0" to="832911536#1" end="1200.00" number="1"/>'),
                (2400.0, '    <flow id="f_train2" type="DEFAULT_RAILTYPE" begin="2400.00" from="832911536#0" to="832911536#1" end="2400.00" number="1"/>'),
                (3600.0, '    <flow id="f_train3" type="DEFAULT_RAILTYPE" begin="3600.00" from="832911536#0" to="832911536#1" end="3600.00" number="1"/>')
            ]
            flows.extend(train_flows)
            total_vehicles += 3  # Add 3 trains
            
            # Validate total vehicles
            if total_vehicles < min_vehicles:
                print(f"Attempt {attempt + 1}: Generated {total_vehicles} vehicles (below threshold of {min_vehicles}), regenerating...")
                continue
            
            print(f"Generated {total_vehicles} vehicles (above threshold of {min_vehicles})")
            
            # Sort flows by departure time
            flows.sort(key=lambda x: x[0])
            
            # Add sorted flows to route file content
            for _, flow_content in flows:
                route_file_content += flow_content + '\n'
            
            route_file_content += "</routes>"
            
            # Write to temporary route file
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
    def __init__(self, sumocfg_file, max_steps=1800, measurement_interval=10, use_gui=True, num_vehicles=50, fast_mode=False):
        """Initialize the traffic environment."""
        self.sumocfg_file = sumocfg_file
        self.max_steps = max_steps
        self.measurement_interval = measurement_interval
        self.use_gui = use_gui
        self.num_vehicles = num_vehicles
        self.fast_mode = fast_mode
        self.steps = 0
        self.done = False
        
        # Initialize dictionaries and metrics
        self.junctions = {}
        self.edges = {}
        self.edge_lengths = {}
        self.edge_speeds = {}
        self.edge_lanes = {}
        self.controllable_junctions = {}
        self.phases = {}
        self.rail_edges = []
        self.rail_lookahead_edges = []
        
        # Initialize junction_actions and state_size (will be set after network initialization)
        self.junction_actions = {}
        self.state_size = 49  # Default state size
        
        # Initialize TensorBoard writer for logging
        try:
            self.writer = SummaryWriter(f'runs/traffic_env_{time.time()}')
        except:
            # If TensorBoard is not available, create a dummy writer
            class DummyWriter:
                def add_scalar(self, *args, **kwargs):
                    pass
            self.writer = DummyWriter()
        
        # Initialize metrics
        self._initialize_metrics()
        
        # Connection and simulation parameters
        self.port = None
        self.sumo_process = None
        self.connection = None
        self.min_phase_duration = 30  # Minimum phase duration in seconds
        
        # Initialize SUMO connection
        self._initialize_sumo()
        
        # Initialize network after SUMO connection is established
        self._initialize_network()

    def _initialize_sumo(self):
        """Initialize SUMO connection with proper error handling."""
        max_attempts = 3
        base_port = 8000
        
        # Try to find SUMO using system PATH first
        sumo_binary = None
        successful_path = None
        
        # Method 1: Try using 'sumo' command from PATH
        try:
            import shutil
            sumo_path = shutil.which('sumo')
            if sumo_path:
                print(f"Found SUMO in PATH: {sumo_path}")
                if os.path.exists(sumo_path):
                    sumo_binary = sumo_path
                    successful_path = sumo_path
                    print(f"Using SUMO from PATH: {successful_path}")
        except Exception as e:
            print(f"Could not find SUMO in PATH: {e}")
        
        # Method 2: If PATH method failed, try hardcoded paths
        if sumo_binary is None:
            # Define SUMO paths - adjust these paths based on your SUMO installation
            sumo_paths = [
                r"C:\Program Files (x86)\Eclipse\Sumo\bin\sumo.exe",  # Default Windows installation
                r"C:\Program Files\Eclipse\Sumo\bin\sumo.exe",        # Alternative Windows installation
                r"C:\Sumo\bin\sumo.exe",                              # Custom installation
                r"C:\Program Files (x86)\Eclipse\Sumo\bin\sumo-gui.exe",  # GUI version
                r"C:\Program Files\Eclipse\Sumo\bin\sumo-gui.exe",        # GUI version alternative
                r"C:\Sumo\bin\sumo-gui.exe"                           # GUI version custom
            ]
            
            if self.use_gui:
                for path in sumo_paths:
                    gui_path = path.replace("sumo.exe", "sumo-gui.exe")
                    print(f"Checking GUI path: {gui_path}")
                    if os.path.exists(gui_path):
                        sumo_binary = gui_path
                        successful_path = gui_path
                        print(f"Found SUMO GUI at: {successful_path}")
                        break
            else:
                for path in sumo_paths:
                    print(f"Checking path: {path}")
                    if os.path.exists(path):
                        sumo_binary = path
                        successful_path = path
                        print(f"Found SUMO at: {successful_path}")
                        break
        
        if sumo_binary is None:
            raise Exception("Could not find SUMO executable. Please ensure SUMO is installed and the path is correct.")
        
        print(f"Using SUMO binary: {sumo_binary}")
        
        for attempt in range(max_attempts):
            try:
                # Clean up any existing connections
                self._cleanup_connection()
                
                # Calculate port number
                self.port = base_port + attempt
                
                # Prepare SUMO command as a list for traci.start
                sumo_cmd = [
                    sumo_binary,
                    "-c", self.sumocfg_file,
                    "--no-step-log", "true",
                    "--no-warnings", "true",
                    "--quit-on-end", "true"
                ]
                
                # Add performance optimizations
                if self.fast_mode:
                    sumo_cmd.extend([
                        "--time-to-teleport", "60",  # Faster teleporting
                        "--collision.action", "none",  # Disable collision checking
                        "--collision.stoptime", "0",
                        "--collision.check-junctions", "0",
                        "--ignore-junction-blocker", "1",
                        "--no-internal-links", "1",
                        "--no-step-log", "true",
                        "--no-warnings", "true",
                        "--error-log", "sumo_errors.log",
                        "--log", "sumo.log",
                        "--tripinfo-output", "tripinfo.xml",
                        "--summary-output", "summary.xml",
                        "--tripinfo-output.write-unfinished", "false",
                        "--vehroute-output", "vehroutes.xml",
                        "--vehroute-output.route-length", "false",
                        "--vehroute-output.speed", "false",
                        "--vehroute-output.waiting-time", "false",
                        "--vehroute-output.accelerations", "false",
                        "--vehroute-output.decelerations", "false",
                        "--vehroute-output.reroute", "false",
                        "--vehroute-output.speeddev", "false",
                        "--vehroute-output.arrival", "false",
                        "--vehroute-output.depart", "false",
                        "--vehroute-output.departLane", "false",
                        "--vehroute-output.departPos", "false",
                        "--vehroute-output.departSpeed", "false",
                        "--vehroute-output.arrivalLane", "false",
                        "--vehroute-output.arrivalPos", "false",
                        "--vehroute-output.arrivalSpeed", "false",
                        "--vehroute-output.stops", "false",
                        "--vehroute-output.signals", "false",
                        "--vehroute-output.moveReminders", "false",
                        "--vehroute-output.ended", "false",
                        "--vehroute-output.start", "false",
                        "--vehroute-output.end", "false",
                        "--vehroute-output.time", "false",
                        "--vehroute-output.duration", "false",
                        "--vehroute-output.route", "false",
                        "--vehroute-output.edges", "false",
                        "--vehroute-output.vType", "false",
                        "--vehroute-output.color", "false",
                        "--vehroute-output.waitSteps", "false",
                        "--vehroute-output.vehCount", "false",
                        "--vehroute-output.personCount", "false",
                        "--vehroute-output.containerCount", "false",
                        "--vehroute-output.teleports", "false",
                        "--vehroute-output.teleportReason", "false",
                        "--vehroute-output.teleportCount", "false",
                        "--vehroute-output.teleportTime", "false",
                        "--vehroute-output.teleportFrom", "false",
                        "--vehroute-output.teleportTo", "false",
                        "--vehroute-output.teleportSpeed", "false",
                        "--vehroute-output.teleportSpeedDev", "false",
                        "--vehroute-output.teleportAccel", "false",
                        "--vehroute-output.teleportDecel", "false",
                        "--vehroute-output.teleportReason", "false",
                        "--vehroute-output.teleportCount", "false",
                        "--vehroute-output.teleportTime", "false",
                        "--vehroute-output.teleportFrom", "false",
                        "--vehroute-output.teleportTo", "false",
                        "--vehroute-output.teleportSpeed", "false",
                        "--vehroute-output.teleportSpeedDev", "false",
                        "--vehroute-output.teleportAccel", "false",
                        "--vehroute-output.teleportDecel", "false"
                    ])
                else:
                    # Standard mode with basic optimizations
                    sumo_cmd.extend([
                        "--time-to-teleport", "300",
                        "--collision.action", "none",
                        "--collision.stoptime", "0",
                        "--collision.check-junctions", "0",
                        "--ignore-junction-blocker", "1",
                        "--no-internal-links", "1"
                    ])
                
                print(f"Starting SUMO on port {self.port}...")
                print(f"SUMO command: {' '.join(sumo_cmd)}")
                
                # Start SUMO using traci.start with list command
                traci.start(sumo_cmd)
                
                # Wait a moment for SUMO to start
                time.sleep(2)
                
                # Verify connection and step simulation once
                try:
                    # Check if TraCI is connected by trying to get simulation time
                    traci.simulation.getTime()
                    # Step simulation once to initialize network
                    traci.simulationStep()
                    print(f"Successfully connected to SUMO on port {self.port}")
                    return
                except:
                    raise Exception("TraCI connection not established")
                
            except Exception as e:
                print(f"Attempt {attempt + 1} failed to start SUMO: {str(e)}")
                self._cleanup_connection()
                if attempt < max_attempts - 1:
                    time.sleep(2)  # Wait before retrying
                continue
        
        raise Exception("Failed to initialize SUMO after multiple attempts")

    def _initialize_network(self):
        """Initialize the traffic network and junction information."""
        try:
            # Check if TraCI is connected by trying to get simulation time
            traci.simulation.getTime()
        except:
            raise Exception("Cannot initialize network: No active SUMO connection")
        
        try:
            print("Initializing network...")
            
            # Ensure simulation is running
            if traci.simulation.getTime() == 0:
                traci.simulationStep()
            
            # Get all junctions
            junction_ids = traci.junction.getIDList()
            for junction_id in junction_ids:
                self.junctions[junction_id] = junction_id
                self.controllable_junctions[junction_id] = {
                    "phases": [],
                    "current_phase": 0,
                    "last_change": 0
                }
            
            # Get all edges
            edge_ids = traci.edge.getIDList()
            print(f"Found {len(edge_ids)} edges in the network")
            for edge_id in edge_ids:
                try:
                    # Get edge attributes using traci directly
                    # First ensure the edge exists in the simulation
                    if edge_id in traci.edge.getIDList():
                        # Get attributes with proper error handling
                        try:
                            length = float(traci.edge.getLength(edge_id))
                        except:
                            length = 100.0  # Default length
                            
                        try:
                            speed = float(traci.edge.getMaxSpeed(edge_id))
                        except:
                            speed = 13.89  # Default speed (50 km/h)
                            
                        try:
                            lanes = int(traci.edge.getLaneNumber(edge_id))
                        except:
                            lanes = 1  # Default number of lanes
                        
                        # Store edge information
                        self.edges[edge_id] = {
                            'id': edge_id,
                            'length': length,
                            'speed': speed,
                            'lanes': lanes
                        }
                        self.edge_lengths[edge_id] = length
                        self.edge_speeds[edge_id] = speed
                        self.edge_lanes[edge_id] = lanes
                    else:
                        # Edge not found in simulation, use defaults
                        self.edges[edge_id] = {
                            'id': edge_id,
                            'length': 100.0,
                            'speed': 13.89,
                            'lanes': 1
                        }
                        self.edge_lengths[edge_id] = self.edges[edge_id]['length']
                        self.edge_speeds[edge_id] = self.edges[edge_id]['speed']
                        self.edge_lanes[edge_id] = self.edges[edge_id]['lanes']
                        
                except Exception as e:
                    print(f"Warning: Could not get attributes for edge {edge_id}: {e}")
                    # Set default values
                    self.edges[edge_id] = {
                        'id': edge_id,
                        'length': 100.0,
                        'speed': 13.89,
                        'lanes': 1
                    }
                    self.edge_lengths[edge_id] = self.edges[edge_id]['length']
                    self.edge_speeds[edge_id] = self.edges[edge_id]['speed']
                    self.edge_lanes[edge_id] = self.edges[edge_id]['lanes']
            
            # Get traffic light phases
            tl_ids = traci.trafficlight.getIDList()
            for junction_id in list(self.controllable_junctions.keys()):
                try:
                    if junction_id in tl_ids:
                        program = traci.trafficlight.getAllProgramLogics(junction_id)[0]
                        self.phases[junction_id] = program.phases
                        self.controllable_junctions[junction_id]["phases"] = program.phases
                    else:
                        del self.controllable_junctions[junction_id]
                except Exception as e:
                    print(f"Warning: Could not get phases for junction {junction_id}: {e}")
                    del self.controllable_junctions[junction_id]
            
            # Initialize rail-related attributes
            self.rail_edges = [edge for edge in self.edges if 'rail' in edge.lower()]
            self.rail_lookahead_edges = self.rail_edges[:2] if self.rail_edges else []
            
            if not self.controllable_junctions:
                raise Exception("No controllable junctions found in the network")
            
            # Populate junction_actions dictionary with number of phases for each junction
            for junction_id in self.controllable_junctions:
                if junction_id in self.phases:
                    self.junction_actions[junction_id] = len(self.phases[junction_id])
                else:
                    self.junction_actions[junction_id] = 4  # Default to 4 phases if not found
            
            print("Network initialization complete")
            print(f"Found {len(self.controllable_junctions)} controllable junctions")
            print(f"Found {len(self.edges)} edges in total")
            print(f"Junction actions: {self.junction_actions}")
            
        except Exception as e:
            print(f"Error initializing network: {e}")
            self._cleanup_connection()
            raise

    def _cleanup_connection(self):
        """Clean up any existing SUMO connection and processes"""
        try:
            # Try to close any existing traci connection
            try:
                traci.close()
            except:
                pass
            
            # Kill any existing SUMO processes
            if os.name == 'nt':  # Windows
                os.system('taskkill /f /im sumo-gui.exe >nul 2>&1')
                os.system('taskkill /f /im sumo.exe >nul 2>&1')
            else:  # Unix-like
                os.system('pkill -f sumo-gui')
                os.system('pkill -f sumo')
            
            # Wait for processes to fully terminate
            time.sleep(2)
            
            # Reset connection flags
            self.connection_closed = True
            self.connection_initialized = False
            
        except Exception as e:
            print(f"Warning during cleanup: {e}")

    def reset(self, route_file=None):
        # Reset step counter for new episode
        self.steps = 0
        
        # Reset episode metrics for new episode
        self._initialize_metrics()
        
        # First, clean up any existing connections
        self._cleanup_connection()
        
        # Generate new route file if none provided
        if route_file is None:
            route_file = generate_new_random_traffic()
        
        # Start SUMO with the specified route file
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # Ensure no existing SUMO processes
                self._cleanup_connection()
                
                # Use the same SUMO binary that was found in _initialize_sumo
                import shutil
                sumo_binary = shutil.which('sumo')
                if not sumo_binary:
                    # Fallback to hardcoded paths
                    sumo_paths = [
                        r"C:\Program Files (x86)\Eclipse\Sumo\bin\sumo.exe",
                        r"C:\Program Files\Eclipse\Sumo\bin\sumo.exe",
                        r"C:\Sumo\bin\sumo.exe"
                    ]
                    for path in sumo_paths:
                        if os.path.exists(path):
                            sumo_binary = path
                            break
                
                if not sumo_binary:
                    raise Exception("Could not find SUMO executable")
                
                # Prepare SUMO command using the found binary
                sumo_cmd = [sumo_binary, "-c", self.sumocfg_file]
                if not self.use_gui:
                    sumo_cmd.extend(["--no-step-log", "true", "--no-warnings", "true"])
                sumo_cmd.extend(["--route-files", route_file])
                
                # Add additional SUMO options for stability
                sumo_cmd.extend([
                    "--time-to-teleport", "300",
                    "--collision.action", "none",
                    "--collision.stoptime", "0",
                    "--collision.check-junctions", "0",
                    "--ignore-junction-blocker", "1",
                    "--no-internal-links", "1"
                ])
                
                # Start SUMO with a new connection
                traci.start(sumo_cmd)
                self.connection_closed = False
                self.connection_initialized = True
                
                # Wait for SUMO to initialize
                time.sleep(1)
                
                # Verify connection
                try:
                    traci.simulation.getTime()
                    return self._get_state()
                except:
                    raise Exception("SUMO connection not established")
                    
            except Exception as e:
                print(f"Attempt {retry_count + 1} failed to start SUMO: {e}")
                retry_count += 1
                time.sleep(2)  # Wait before retrying
                
                # Clean up before next attempt
                self._cleanup_connection()

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
            # Check if TraCI is connected and simulation is loaded
            if not traci.isLoaded():
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
                # Use a simpler approach - just get all edges and use them
                # This avoids the complex junction-edge mapping that was causing errors
                edges_to_use = list(self.edges.keys())[:4]  # Use first 4 edges
                
                for edge in edges_to_use:
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
                remaining_slots = 4 - min(len(edges_to_use), 4)
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
            # Iterate over actual edges, not junction-edge mappings
            for edge_id in self.edges.keys():
                try:
                    # Get number of vehicles and their waiting times
                    vehicles = traci.edge.getLastStepVehicleNumber(edge_id)
                    waiting_time = traci.edge.getWaitingTime(edge_id)
                    
                    # Consider a vehicle as queued if it has waiting time > 0
                    if waiting_time > 0:
                        total_queues[edge_id] = vehicles
                        total_waiting_time += waiting_time
                except traci.exceptions.TraCIException:
                    continue
        
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
            # Iterate over actual edges, not junction-edge mappings
            for edge_id in self.edges.keys():
                try:
                    vehicles = traci.edge.getLastStepVehicleIDs(edge_id)
                    for vid in vehicles:
                        speed = traci.vehicle.getSpeed(vid)
                        if speed > MIN_SPEED_THRESHOLD:
                            speed_mph = speed * 2.237  # Convert m/s to mph
                            total_speed += speed_mph
                            total_moving_vehicles += 1
                            
                            # Log per-vehicle speeds
                            self.writer.add_scalar(f'Edge_{edge_id}/Vehicle_{vid}/Speed', 
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
            
            # Iterate over actual edges, not junction-edge mappings
            for edge_id in self.edges.keys():
                try:
                    vehicles = traci.edge.getLastStepVehicleIDs(edge_id)
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
            # Iterate over actual edges, not junction-edge mappings
            for edge_id in self.edges.keys():
                try:
                    vehicles = traci.edge.getLastStepVehicleIDs(edge_id)
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
        total_queue = 0
        total_lanes = 0
        critical_lanes = 0
        
        try:
            # Iterate over actual edges, not junction-edge mappings
            for edge_id in self.edges.keys():
                try:
                    waiting_count = traci.edge.getLastStepHaltingNumber(edge_id)
                    queue_lengths[edge_id] = waiting_count
                    
                    # Track critical lanes
                    if waiting_count > performance_targets['lane_thresholds']['critical']:
                        critical_lanes += 1
                    
                    total_queue += waiting_count
                    total_lanes += 1
                except traci.exceptions.TraCIException:
                    continue
            
            # Calculate average queue per lane
            avg_queue_per_lane = total_queue / max(total_lanes, 1)
            critical_lane_ratio = critical_lanes / max(total_lanes, 1)
            
            # Log aggregate metrics
            self.writer.add_scalar('Metrics/Average_Queue_Per_Lane', 
                                 avg_queue_per_lane, self.steps)
            self.writer.add_scalar('Metrics/Critical_Lanes_Ratio', 
                                 critical_lane_ratio, self.steps)
            
            return queue_lengths, avg_queue_per_lane, {}, critical_lane_ratio
        except Exception as e:
            print(f"Error getting queue lengths: {e}")
            return {}, 0, {}, 0

    def _update_metrics(self):
        """Update traffic metrics for the current simulation step"""
        try:
            # In fast mode, update metrics less frequently to save time
            if self.fast_mode and self.steps % 50 != 0:  # Update every 50 steps in fast mode
                return
            elif not self.fast_mode and self.steps % 10 != 0:  # Update every 10 steps in standard mode
                return
            
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
            
            # Only print metrics when the episode is actually done
            if self.done:
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
        """Execute one step in the environment."""
        try:
            # Convert action to dictionary format if it's a tuple or integer
            if isinstance(action, (tuple, int)):
                action_dict = {}
                if isinstance(action, tuple):
                    for i, junction_name in enumerate(self.controllable_junctions.keys()):
                        action_dict[junction_name] = {
                            'phase': action[i] % len(self.phases[junction_name]),
                            'duration': self.min_phase_duration
                        }
                else:  # action is an integer
                    for junction_name in self.controllable_junctions.keys():
                        action_dict[junction_name] = {
                            'phase': action % len(self.phases[junction_name]),
                            'duration': self.min_phase_duration
                        }
                action = action_dict
            
            # Execute actions for each junction
            for junction_name, junction_action in action.items():
                if junction_name in self.controllable_junctions:
                    try:
                        # Get the phase index and ensure it's within bounds
                        phase_idx = junction_action['phase']
                        if not isinstance(phase_idx, int):
                            phase_idx = int(phase_idx)
                        phase_idx = phase_idx % len(self.phases[junction_name])
                        
                        # Set the phase and duration
                        traci.trafficlight.setPhase(junction_name, phase_idx)
                        traci.trafficlight.setPhaseDuration(junction_name, junction_action['duration'])
                        
                        # Update junction state
                        self.controllable_junctions[junction_name]["current_phase"] = phase_idx
                        self.controllable_junctions[junction_name]["last_change"] = traci.simulation.getTime()
                    except Exception as e:
                        print(f"Warning: Could not set phase for junction {junction_name}: {e}")
                        continue
            
            # Step the simulation
            traci.simulationStep()
            
            # Update metrics and calculate reward
            self._update_metrics()
            reward = self._calculate_reward()
            
            # Increment step counter first
            self.steps += 1
            
            # Get new state and check if episode is done
            state = self._get_state()
            done = self.steps >= self.max_steps
            
            return state, reward, done, {}
            
        except traci.exceptions.TraCIException as e:
            print(f"TraCI error during step: {e}")
            return self._get_state(), -100, True, {'error': str(e)}
        except Exception as e:
            print(f"Error during step: {e}")
            return self._get_state(), -100, True, {'error': str(e)}

    def _calculate_reward(self):
        try:
            # Check if all required metrics are available
            if not (self.episode_metrics['waiting_times'] and
                    self.episode_metrics['speeds'] and
                    self.episode_metrics['queue_lengths'] and
                    self.episode_metrics['travel_times'] and
                    self.episode_metrics['traffic_stats']):
                return 0.0

            import os
            reward_variant = os.getenv("REWARD_VARIANT", "baseline")

            # Retrieve current metrics
            current_waiting_time = self.episode_metrics['waiting_times'][-1]
            current_speed = self.episode_metrics['speeds'][-1]
            current_queue = self.episode_metrics['queue_lengths'][-1]
            current_travel_time = self.episode_metrics['travel_times'][-1]
            traffic_stats = self.episode_metrics['traffic_stats'][-1]
            stopped_vehicles = traffic_stats['stopped_vehicles']

            # Default reward (baseline)
            reward = (
                -0.4 * current_waiting_time
                -0.3 * current_queue
                + 1.0 * current_speed
                -0.2 * current_travel_time
                -0.2 * stopped_vehicles
            )

            # Variant-specific reward definitions
            if reward_variant == "safety":
                safety_penalty = 0
                if self.train_approaching:
                    safety_penalty = 5 * current_queue + 3 * stopped_vehicles
                reward = (
                    -0.3 * current_waiting_time
                    -0.5 * current_queue
                    + 1.0 * current_speed
                    -0.2 * current_travel_time
                    -0.2 * stopped_vehicles
                    - safety_penalty
                )

            elif reward_variant == "queue":
                reward = (
                    -0.2 * current_waiting_time
                    -0.7 * current_queue
                    + 1.0 * current_speed
                    -0.1 * current_travel_time
                    -0.1 * stopped_vehicles
                )

            elif reward_variant == "eco":
                eco_smoothness = -0.3 * stopped_vehicles
                reward = (
                    -0.3 * current_waiting_time
                    -0.3 * current_queue
                    + 1.0 * current_speed
                    -0.2 * current_travel_time
                    + eco_smoothness
                )

            elif reward_variant == "trainaware":
                train_weight = 0.0
                if self.train_distance < 200:
                    train_weight = 5.0
                reward = (
                    -0.3 * current_waiting_time
                    -0.3 * current_queue
                    + 1.0 * current_speed
                    -0.3 * current_travel_time
                    - train_weight * current_queue
                )

            reward /= max(self.total_vehicles_passed, 1)
            reward = float(np.clip(reward, -100, 100))

            return reward

        except Exception as e:
            print(f"Reward error: {e}")
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

    def close(self):
        """Properly close the environment."""
        self._cleanup_connection()

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
        # Get next Q values for each junction (Double DQN)
        with torch.no_grad():
            next_policy_q = self.policy_net(next_state_batch)
            next_target_q = self.target_net(next_state_batch)
            next_q_values = {}
            for junction in self.junction_actions:
                # Double DQN: use online policy to select action, but target to evaluate
                next_policy_logits = next_policy_q[junction]['phase_logits']
                best_next_actions = next_policy_logits.argmax(1, keepdim=True)
                next_q_values[junction] = next_target_q[junction]['phase_logits'].gather(1, best_next_actions)
        
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

    def load(self, path):
        """Load model weights from file"""
        try:
            # Load the entire checkpoint with weights_only=False
            checkpoint = torch.load(path, weights_only=False, map_location=self.device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.policy_net.load_state_dict(checkpoint['model_state_dict'])
                if 'optimizer_state_dict' in checkpoint:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print(f"Loaded checkpoint from {path}")
            else:
                # Fallback for older checkpoint format
                self.policy_net.load_state_dict(checkpoint)
                print(f"Loaded model weights from {path}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            raise

    @property
    def epsilon(self):
        """Get current epsilon value for exploration"""
        return self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.steps_done / self.eps_decay)

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
    # Add TRB-formatted plot for this episode
    plot_training_metrics_trb_style(
        rewards_history, waiting_times_history, avg_speeds_history,
        queue_lengths_history, travel_times_history,
        epsilon_history, stopped_vehicles_history, episode_dir
    )
    
    print(f"\nSaved episode {episode+1} plots to: {episode_dir}")

def train(episodes=150, min_episodes=75, patience=50, target_reward=75, fast_mode=False, hybrid_mode=False, disable_plots=False):
    """Train the agent with performance-focused parameters."""
    print("\nStarting training with performance-focused parameters...")
    print(f"Fast mode: {'Enabled' if fast_mode else 'Disabled'}")
    print(f"Hybrid mode: {'Enabled' if hybrid_mode else 'Disabled'}")
    print(f"Plotting: {'Disabled' if disable_plots else 'Enabled'}")
    print("\nSimulation Duration: 1800 seconds (30 minutes) per episode")
    
    # Performance optimization settings
    if fast_mode:
        # Fast mode: reduce simulation time and frequency of expensive operations
        simulation_steps = 600  # 10 minutes instead of 30 (3x faster)
        progress_update_freq = 300  # Update progress less frequently
        metric_update_freq = 50  # Update metrics much less frequently
        print(f"Fast mode: Using {simulation_steps} steps per episode (10 minutes)")
        print(f"Fast mode: Metrics updated every {metric_update_freq} steps")
    elif hybrid_mode:
        # Hybrid mode: keep full simulation time but optimize other aspects
        simulation_steps = 1800  # Full 30 minutes
        progress_update_freq = 200  # Update progress less frequently
        metric_update_freq = 30  # Update metrics less frequently
        print(f"Hybrid mode: Using {simulation_steps} steps per episode (30 minutes)")
        print(f"Hybrid mode: Metrics updated every {metric_update_freq} steps")
        print(f"Hybrid mode: Full simulation time with speed optimizations")
    else:
        simulation_steps = 1800  # Full 30 minutes
        progress_update_freq = 100  # Update progress every 100 steps
        metric_update_freq = 10  # Update metrics every 10 steps
        print(f"Standard mode: Using {simulation_steps} steps per episode (30 minutes)")
    
    # Create output directories for this variant
    variant = os.environ.get('REWARD_VARIANT', 'baseline')
    output_dir = f"output_{variant}"
    plots_dir = os.path.join(output_dir, "plots")
    models_dir = os.path.join(output_dir, "models")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    # Initialize metrics history
    rewards_history = []
    waiting_times_history = []
    avg_speeds_history = []
    queue_lengths_history = []
    travel_times_history = []
    epsilon_history = []
    stopped_vehicles_history = []
    
    # Initialize running metrics
    running_reward = 0.0
    running_waiting_time = 0.0
    running_speed = 0.0
    running_travel_time = 0.0
    running_queue_length = 0.0
    running_stopped_vehicles = 0.0
    
    # Initialize environment and agent
    env = None
    try:
        # In hybrid mode, we want speed optimizations but full simulation time
        use_fast_optimizations = fast_mode or hybrid_mode
        env = TrafficEnvironment(sumocfg_file, max_steps=simulation_steps, use_gui=False, fast_mode=use_fast_optimizations)
        state_size = env.state_size
        junction_actions = env.junction_actions
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        agent = DQNAgent(state_size, junction_actions, device)
        
        # Pre-generate route files for faster episode resets
        if fast_mode or hybrid_mode:
            print("Pre-generating route files for optimized mode...")
            route_files = []
            for i in range(min(10, episodes)):  # Pre-generate up to 10 route files
                route_file = generate_new_random_traffic()
                if route_file:
                    route_files.append(route_file)
            print(f"Pre-generated {len(route_files)} route files")
        else:
            route_files = None
        
        print("\n=== Starting Training ===")
        print(f"Training for {episodes} episodes")
        print(f"Minimum episodes: {min_episodes}")
        print(f"Patience: {patience}")
        print(f"Target reward: {target_reward}")
        
        best_reward = float('-inf')
        episodes_without_improvement = 0
        best_agent_state = None
        
        for episode in range(episodes):
            try:
                # Use pre-generated route file if available
                if fast_mode or hybrid_mode and route_files and episode < len(route_files):
                    route_file = route_files[episode]
                else:
                    route_file = None
                
                state = env.reset(route_file=route_file)
                episode_reward = 0
                steps = 0
                
                print(f"\nEpisode {episode + 1}/{episodes} - Starting...")
                
                while True:
                    # Get action from agent
                    action_dict = agent.select_action(state)
                    
                    # The agent already returns the action in the correct format
                    # No need to convert - just use it directly
                    action = action_dict
                    
                    # Take step in environment
                    next_state, reward, done, _ = env.step(action)
                    
                    # Update episode metrics (use the reward from the step)
                    episode_reward += reward
                    
                    # Store transition in memory
                    agent.memory.push(state, action_dict, next_state, reward, done)
                    state = next_state
                    steps += 1
                    
                    # Show progress at configurable frequency
                    if steps % progress_update_freq == 0:
                        progress = (steps / env.max_steps) * 100
                        print(f"\rEpisode {episode + 1}/{episodes} - Progress: {progress:.1f}% ({steps}/{env.max_steps})", end='', flush=True)
                    
                    if done:
                        # Show final progress
                        progress = (steps / env.max_steps) * 100
                        print(f"\rEpisode {episode + 1}/{episodes} - Progress: {progress:.1f}% ({steps}/{env.max_steps}) - Complete!")
                        break
                
                # Calculate episode averages
                steps = max(steps, 1)  # Avoid division by zero
                avg_reward = episode_reward / steps
                
                # Get metrics from environment's episode_metrics
                if env.episode_metrics['waiting_times']:
                    avg_waiting_time = np.mean(env.episode_metrics['waiting_times'])
                else:
                    avg_waiting_time = 0.0
                    
                if env.episode_metrics['speeds']:
                    avg_speed = np.mean(env.episode_metrics['speeds'])
                else:
                    avg_speed = 0.0
                    
                if env.episode_metrics['travel_times']:
                    avg_travel_time = np.mean(env.episode_metrics['travel_times'])
                else:
                    avg_travel_time = 0.0
                    
                if env.episode_metrics['queue_lengths']:
                    avg_queue_length = np.mean(env.episode_metrics['queue_lengths'])
                else:
                    avg_queue_length = 0.0
                    
                # Calculate stopped vehicles from traffic stats
                if env.episode_metrics['traffic_stats']:
                    avg_stopped_vehicles = np.mean([stats['stopped_vehicles'] for stats in env.episode_metrics['traffic_stats']])
                else:
                    avg_stopped_vehicles = 0.0
                
                # Update running averages (95/5 weighting)
                running_reward = 0.95 * running_reward + 0.05 * avg_reward
                running_waiting_time = 0.95 * running_waiting_time + 0.05 * avg_waiting_time
                running_speed = 0.95 * running_speed + 0.05 * avg_speed
                running_travel_time = 0.95 * running_travel_time + 0.05 * avg_travel_time
                running_queue_length = 0.95 * running_queue_length + 0.05 * avg_queue_length
                running_stopped_vehicles = 0.95 * running_stopped_vehicles + 0.05 * avg_stopped_vehicles
                
                # Store metrics in history
                rewards_history.append(running_reward)
                waiting_times_history.append(running_waiting_time)
                avg_speeds_history.append(running_speed)
                queue_lengths_history.append(running_queue_length)
                travel_times_history.append(running_travel_time)
                epsilon_history.append(agent.epsilon)
                stopped_vehicles_history.append(running_stopped_vehicles)
                
                # Print episode summary
                print(f"\nEpisode {episode + 1}/{episodes}")
                print(f"Steps: {steps}")
                print(f"Average Reward: {avg_reward:.2f}")
                print(f"Running Reward: {running_reward:.2f}")
                print(f"Average Waiting Time: {avg_waiting_time:.2f}s")
                print(f"Average Speed: {avg_speed:.2f} mph")
                print(f"Average Queue Length: {avg_queue_length:.2f}")
                print(f"Average Travel Time: {avg_travel_time:.2f}s")
                print(f"Stopped Vehicles: {avg_stopped_vehicles:.2f}")
                print(f"Epsilon: {agent.epsilon:.3f}")
                
                # Save model if it's the best so far
                if running_reward > best_reward:
                    best_reward = running_reward
                    best_agent_state = agent.policy_net.state_dict()
                    agent.save(os.path.join(models_dir, f"best_model_{variant}.pth"))
                    episodes_without_improvement = 0
                else:
                    episodes_without_improvement += 1
                
                # Save progress every 10 episodes
                if (episode + 1) % 10 == 0 and not disable_plots:
                    agent.save(os.path.join(models_dir, f"model_{variant}_episode_{episode + 1}.pth"))
                    save_episode_plots(
                        episode + 1, plots_dir,
                        rewards_history, waiting_times_history,
                        avg_speeds_history, queue_lengths_history,
                        travel_times_history, epsilon_history,
                        stopped_vehicles_history
                    )
                elif (episode + 1) % 10 == 0 and disable_plots:
                    # Just save model without plots
                    agent.save(os.path.join(models_dir, f"model_{variant}_episode_{episode + 1}.pth"))
                
                # Early stopping if we've reached the target reward
                if episode >= min_episodes and running_reward >= target_reward:
                    print(f"\nTarget reward of {target_reward} achieved!")
                    break
                
                # Early stopping if no improvement for too long
                if episode >= min_episodes and episodes_without_improvement >= patience:
                    print(f"\nNo improvement for {patience} episodes. Stopping training.")
                    break
                
            except Exception as e:
                print(f"Error during episode {episode}: {str(e)}")
                continue
        
        # Save final model and plots
        if best_agent_state is not None:
            agent.policy_net.load_state_dict(best_agent_state)
            agent.save(os.path.join(models_dir, f"final_model_{variant}.pth"))
        
        if not disable_plots:
            save_episode_plots(
                episodes, plots_dir,
                rewards_history, waiting_times_history,
                avg_speeds_history, queue_lengths_history,
                travel_times_history, epsilon_history,
                stopped_vehicles_history
            )
        else:
            print("Skipping final plots (plotting disabled)")
        
        print(f"\nTraining completed for variant: {variant}")
        print(f"Best reward achieved: {best_reward:.2f}")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
    finally:
        if env is not None:
            env.close()
    
    return rewards_history, waiting_times_history, avg_speeds_history, queue_lengths_history, travel_times_history, epsilon_history, stopped_vehicles_history

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
    episodes = np.arange(1, len(rewards_history) + 1)
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

def plot_training_metrics_trb_style(
    rewards, waiting_times, speeds, queues, travel_times, epsilon, stopped, save_dir
):
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    episodes = np.arange(1, len(rewards) + 1)

    # TRB publication-style settings
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'font.size': 11,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 9,
        'lines.linewidth': 2,
        'figure.figsize': (7.5, 10),
        'axes.spines.top': False,
        'axes.spines.right': False
    })

    fig, axs = plt.subplots(3, 2, figsize=(7.5, 10))
    fig.subplots_adjust(hspace=0.5, wspace=0.3)

    metrics = [
        (rewards, "Episode Reward", "Reward"),
        (queues, "Queue Length", "Vehicles/lane"),
        (stopped, "Stopped Vehicles", "Count"),
        (waiting_times, "Waiting Time", "Seconds"),
        (speeds, "Traffic Speed", "mph"),
        (travel_times, "Travel Time", "Seconds"),
    ]

    for ax, (data, title, ylabel) in zip(axs.flat, metrics):
        ax.plot(episodes, data, color='black', marker='o', markersize=3, linestyle='-')
        ax.set_title(title)
        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        ax.set_xlim(left=1)
        ax.tick_params(direction='in')

    # Save in the same directory as other plots
    save_path = os.path.join(save_dir, 'trb_formatted_metrics.png')
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"TRB-formatted plot saved to: {save_path}")

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
    try:
        parser = argparse.ArgumentParser(description='Train DQN agent for traffic control')
        parser.add_argument('--episodes', type=int, default=150, help='Number of episodes to train')
        parser.add_argument('--min-episodes', type=int, default=75, help='Minimum number of episodes before early stopping')
        parser.add_argument('--patience', type=int, default=50, help='Patience for early stopping')
        parser.add_argument('--target-reward', type=float, default=75, 
                          help='Target reward for early stopping. 75 requires excellent performance: high speeds, low waiting times, and efficient train handling')
        parser.add_argument('--fast-mode', action='store_true', 
                          help='Enable fast mode for quicker training (reduced simulation time and less frequent metric updates)')
        parser.add_argument('--hybrid-mode', action='store_true', 
                          help='Enable hybrid mode for full simulation time but optimized other aspects')
        parser.add_argument('--disable-plots', action='store_true',
                          help='Disable plotting for maximum training speed')
        args = parser.parse_args()
        
        # Ensure any existing SUMO connections are closed
        try:
            traci.close()
        except:
            pass
            
        rewards_history, waiting_times_history, avg_speeds_history, queue_lengths_history, travel_times_history, epsilon_history, stopped_vehicles_history = train(
            episodes=args.episodes,
            min_episodes=args.min_episodes,
            patience=args.patience,
            target_reward=args.target_reward,
            fast_mode=args.fast_mode,
            hybrid_mode=args.hybrid_mode,
            disable_plots=args.disable_plots
        )
        
    except Exception as e:
        print(f"Error: {str(e)}")
        # Ensure SUMO connection is closed on error
        try:
            traci.close()
        except:
            pass
        sys.exit(1)