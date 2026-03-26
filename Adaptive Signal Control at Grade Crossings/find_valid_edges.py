import os
import traci
from sumolib import checkBinary

def find_valid_edges():
    """Find valid edges in the SUMO network"""
    sumo_binary = checkBinary("sumo")
    sumocfg_file = "osm.sumocfg"
    
    # Start SUMO to get edge information
    traci.start([sumo_binary, "-c", sumocfg_file, "--no-step-log", "true", "--no-warnings", "true"])
    
    try:
        # Get all edges
        all_edges = traci.edge.getIDList()
        print(f"Found {len(all_edges)} edges in the network")
        
        # Get some sample edges for routes
        sample_edges = all_edges[:20]  # First 20 edges
        print("Sample valid edges:")
        for i, edge in enumerate(sample_edges):
            print(f"  {i+1}. {edge}")
            
        return sample_edges
        
    finally:
        traci.close()

if __name__ == "__main__":
    valid_edges = find_valid_edges() 