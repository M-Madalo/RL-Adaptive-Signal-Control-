import torch

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define performance targets globally
performance_targets = {
    'waiting_time': 90.0,    # Maximum average waiting time in seconds (upper bound)
    'min_waiting_time': 45.0, # Minimum average waiting time in seconds (lower bound)
    'speed': 40.0,          # Target average speed in mph
    'queue_length': 5.0,     # Maximum average queue length
    'travel_time': 120.0,    # Maximum average travel time in seconds
    'emergency_braking': 2.0 # Maximum average emergency braking events per episode
} 