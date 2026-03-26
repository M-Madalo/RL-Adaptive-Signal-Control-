"""
metrics.py: Metrics and reward computation for traffic RL environment.
"""
import numpy as np
import traci

def get_traffic_stats(env):
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
        STOPPED_THRESHOLD = 0.1
        WAITING_THRESHOLD = 0.5
        for junction_name, edges in env.monitored_edges.items():
            for edge in edges:
                try:
                    vehicles = traci.edge.getLastStepVehicleIDs(edge)
                    for vid in vehicles:
                        speed = traci.vehicle.getSpeed(vid)
                        stats['total_vehicles'] += 1
                        total_speed_all += speed * 2.237
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
        if stats['moving_vehicles'] > 0:
            stats['avg_speed_moving'] = total_speed_moving / stats['moving_vehicles']
        if stats['total_vehicles'] > 0:
            stats['avg_speed_all'] = total_speed_all / stats['total_vehicles']
        return stats
    except Exception as e:
        print(f"Error getting traffic stats: {e}")
        return stats

def get_average_waiting_time(env):
    """Get the average waiting time for vehicles that are either stopped or moving slowly"""
    total_waiting_time = 0
    waiting_vehicles = 0
    WAITING_THRESHOLD = 0.5
    try:
        for junction_name, edges in env.monitored_edges.items():
            for edge in edges:
                try:
                    vehicles = traci.edge.getLastStepVehicleIDs(edge)
                    for veh_id in vehicles:
                        speed = traci.vehicle.getSpeed(veh_id)
                        if speed <= WAITING_THRESHOLD:
                            waiting_time = traci.vehicle.getWaitingTime(veh_id)
                            if waiting_time > 0:
                                total_waiting_time += waiting_time
                                waiting_vehicles += 1
                except traci.exceptions.TraCIException:
                    continue
        return total_waiting_time / max(waiting_vehicles, 1)
    except Exception as e:
        print(f"Error getting average waiting time: {e}")
        return 0

def get_queue_lengths(env):
    """Get detailed queue lengths with critical lane tracking"""
    queue_lengths = {}
    junction_lane_waiting = {}
    total_queue = 0
    total_lanes = 0
    critical_lanes = 0
    try:
        for junction_name, edges in env.monitored_edges.items():
            junction_lane_waiting[junction_name] = {}
            junction_queue = 0
            for edge in edges:
                try:
                    waiting_count = traci.edge.getLastStepHaltingNumber(edge)
                    queue_lengths[edge] = waiting_count
                    junction_lane_waiting[junction_name][edge] = waiting_count
                    if waiting_count > env.performance_targets['lane_thresholds']['critical']:
                        critical_lanes += 1
                    junction_queue += waiting_count
                    total_lanes += 1
                except traci.exceptions.TraCIException:
                    continue
            total_queue += junction_queue
            env.writer.add_scalar(f'Junction_{junction_name}/Queue_Length', junction_queue, env.steps)
        avg_queue_per_lane = total_queue / max(total_lanes, 1)
        critical_lane_ratio = critical_lanes / max(total_lanes, 1)
        env.writer.add_scalar('Metrics/Average_Queue_Per_Lane', avg_queue_per_lane, env.steps)
        env.writer.add_scalar('Metrics/Critical_Lanes_Ratio', critical_lane_ratio, env.steps)
        return queue_lengths, avg_queue_per_lane, junction_lane_waiting, critical_lane_ratio
    except Exception as e:
        print(f"Error getting queue lengths: {e}")
        return {}, 0, {}, 0

def update_metrics(env):
    """Update traffic metrics for the current simulation step"""
    try:
        current_waiting_time = get_average_waiting_time(env)
        traffic_stats = get_traffic_stats(env)
        current_speed = env._measure_traffic_speed()
        queue_lengths, avg_queue_per_lane, junction_lane_waiting, critical_lane_ratio = get_queue_lengths(env)
        env._update_travel_times()
        current_travel_time = env._get_average_travel_time()
        env.episode_metrics['waiting_times'].append(current_waiting_time)
        env.episode_metrics['speeds'].append(current_speed)
        env.episode_metrics['queue_lengths'].append(avg_queue_per_lane)
        env.episode_metrics['travel_times'].append(current_travel_time)
        if 'traffic_stats' not in env.episode_metrics:
            env.episode_metrics['traffic_stats'] = []
        env.episode_metrics['traffic_stats'].append(traffic_stats)
    except traci.exceptions.FatalTraCIError:
        env.done = True
    except Exception as e:
        print(f"Error updating metrics: {e}")
        env.done = True

def get_reward(env):
    try:
        if not env.episode_metrics['waiting_times'] or \
           not env.episode_metrics['speeds'] or \
           not env.episode_metrics['queue_lengths'] or \
           not env.episode_metrics['traffic_stats']:
            return 0.0
        current_waiting_time = env.episode_metrics['waiting_times'][-1]
        current_speed = env.episode_metrics['speeds'][-1]
        avg_queue = env.episode_metrics['queue_lengths'][-1]
        traffic_stats = env.episode_metrics['traffic_stats'][-1]
        stopped_vehicles = traffic_stats['stopped_vehicles']
        prev_waiting = env.episode_metrics['waiting_times'][-2] if len(env.episode_metrics['waiting_times']) > 1 else current_waiting_time
        prev_queue = env.episode_metrics['queue_lengths'][-2] if len(env.episode_metrics['queue_lengths']) > 1 else avg_queue
        prev_stopped = env.episode_metrics['traffic_stats'][-2]['stopped_vehicles'] if len(env.episode_metrics['traffic_stats']) > 1 else stopped_vehicles
        delta_waiting = prev_waiting - current_waiting_time
        delta_queue = prev_queue - avg_queue
        delta_stopped = prev_stopped - stopped_vehicles
        reward = 0
        reward += (delta_waiting * 3.0)
        reward += (delta_queue * 5.0)
        reward += (delta_stopped * 1.0)
        reward += (current_speed / env.performance_targets['speed']['target']) * 10
        if current_waiting_time > env.performance_targets['waiting_time']['critical']:
            reward -= 20
        if avg_queue > env.performance_targets['lane_thresholds']['critical']:
            reward -= 20
        if stopped_vehicles > 15:
            reward -= 10
        for junction_name in env.controllable_junctions:
            try:
                edges = env.monitored_edges.get(junction_name, [])
                current_count = sum([traci.edge.getLastStepVehicleNumber(e) for e in edges])
                prev_count = env.previous_vehicle_counts.get(junction_name, current_count)
                cleared = max(0, prev_count - current_count)
                green_duration = traci.trafficlight.getPhaseDuration(env.junctions[junction_name])
                if green_duration > 0:
                    efficiency = cleared / green_duration
                    efficiency = np.clip(efficiency, 0, 5.0)
                    reward += efficiency * 2.0
                env.previous_vehicle_counts[junction_name] = current_count
            except Exception as e:
                continue
        reward = float(np.clip(reward, -50, 50))
        env.running_reward_baseline = 0.99 * env.running_reward_baseline + 0.01 * reward
        reward -= env.running_reward_baseline
        return reward
    except Exception as e:
        print(f"Reward error: {e}")
        return 0.0 