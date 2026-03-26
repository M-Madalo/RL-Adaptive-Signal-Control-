import traci
import sys

def get_network_info():
    try:
        # Start SUMO
        traci.start(['sumo', '-c', 'osm.sumocfg'])
        
        # Get all traffic light IDs
        tl_ids = traci.trafficlight.getIDList()
        print("\nTraffic Light IDs:", tl_ids)
        
        # For each traffic light, get its controlled lanes
        print("\nControlled Lanes for each Traffic Light:")
        for tl_id in tl_ids:
            try:
                lanes = traci.trafficlight.getControlledLanes(tl_id)
                print(f"\nTraffic Light {tl_id}:")
                print(f"Controlled Lanes: {lanes}")
                
                # Get the edges these lanes belong to
                edges = set()
                for lane in lanes:
                    edge = lane.rsplit('_', 1)[0]  # Remove lane index to get edge ID
                    edges.add(edge)
                print(f"Connected Edges: {list(edges)}")
                
            except Exception as e:
                print(f"Error getting info for {tl_id}: {str(e)}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        traci.close()

if __name__ == "__main__":
    get_network_info() 