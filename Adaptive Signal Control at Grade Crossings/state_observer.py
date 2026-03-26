"""
state_observer.py: State observation and queue checking for traffic RL environment.
"""
import numpy as np
import traci

def get_state(env):
    """Get enhanced state including train information and queue lengths"""
    state = np.zeros(49, dtype=np.float32)
    try:
        if not traci.isLoaded() or traci.simulation.getMinExpectedNumber() <= 0:
            return state
        if env.steps % 5 == 0:
            env._check_train_presence()
        state[42] = float(env.train_approaching)
        state[43] = min(env.train_distance / 1000.0, 1.0)
        state[44] = min(env.train_speed / 10.0, 1.0)
        idx = 0
        for junction_name, junction_id in env.junctions.items():
            edges = env.monitored_edges.get(junction_name, [])
            if not edges:
                continue
            for edge in edges[:4]:
                try:
                    if idx < 42:
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
            remaining_slots = 4 - min(len(edges), 4)
            if remaining_slots > 0 and idx < 42:
                idx += remaining_slots * 4
        return state
    except traci.exceptions.FatalTraCIError:
        env.done = True
        return state
    except Exception:
        return state

def check_queues(env):
    """Check queue lengths at monitored edges"""
    total_queues = {}
    total_waiting_time = 0
    try:
        for junction_name, edges in env.monitored_edges.items():
            junction_queue = 0
            junction_waiting = 0
            for edge in edges:
                try:
                    vehicles = traci.edge.getLastStepVehicleNumber(edge)
                    waiting_time = traci.edge.getWaitingTime(edge)
                    if waiting_time > 0:
                        junction_queue += vehicles
                        junction_waiting += waiting_time
                except traci.exceptions.TraCIException:
                    continue
            total_queues[junction_name] = junction_queue
            total_waiting_time += junction_waiting
    except traci.exceptions.FatalTraCIError:
        env.done = True
        return {}, 0
    return total_queues, total_waiting_time 