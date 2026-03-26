"""
traffic_network.py: Network setup and SUMO logic for traffic RL environment.
"""
import os
import sys
import traci

class TrafficNetwork:
    def __init__(self, sumocfg_file, use_gui=True):
        self.sumocfg_file = sumocfg_file
        self.use_gui = use_gui
        self.connection_initialized = True
        self.connection_closed = False
        self._initialize_network()

    def _initialize_network(self):
        """Initialize the traffic network configuration"""
        try:
            # Define junction IDs
            self.junctions = {
                "junction1": "202339061",
                "junction2": "202339032",
                "junction3": "202339043",
                "junction4": "202339017",
                "junction5": "202339039",
                "rail_crossing": "202291997"
            }
            # Define monitored edges for each junction
            self.monitored_edges = {
                "junction1": ["245915228#1", "-245915228#2", "-19468172"],
                "junction2": ["340808718#10.50", "683562137#2", "-1346082739", "-634020683"],
                "junction3": ["1347512024", "935204071#1", "-1347512025", "-1347512026"],
                "junction4": ["1347512023#0", "-1347512023#1"],
                "junction5": ["1347512024", "935204071#1", "-1347512025", "-1347512026"],
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
            self.phases = {j: self.controllable_junctions[j]["phases"] for j in self.controllable_junctions}
            self.current_phases = {j: 0 for j in self.controllable_junctions}
            self.last_phase_changes = {j: 0 for j in self.controllable_junctions}
            self.phase_durations = {j: {p: 0 for p in range(len(self.controllable_junctions[j]["phases"]))} for j in self.controllable_junctions}
        except Exception as e:
            print(f"Error initializing network: {e}")
            raise

    def start_sumo(self):
        try:
            sumo_cmd = ["sumo-gui" if self.use_gui else "sumo", "-c", self.sumocfg_file]
            if not self.use_gui:
                sumo_cmd.extend(["--no-step-log", "true", "--no-warnings", "true"])
            traci.start(sumo_cmd)
            self.connection_initialized = True
            self.connection_closed = False
        except Exception as e:
            print(f"Error starting SUMO: {e}")
            self.connection_closed = True
            return None 