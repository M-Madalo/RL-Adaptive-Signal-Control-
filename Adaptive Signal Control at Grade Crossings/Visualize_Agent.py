import torch
import os
import time
from reinforcement_learningCA import DQNAgent, TrafficEnvironment
from traffic_components import device
import traci
import sumolib
from torch.utils.tensorboard import SummaryWriter

def evaluate(agent, env, episodes=1):
    """Evaluate the trained agent."""
    agent.eps_start = 0.0
    agent.eps_end = 0.0
    agent.eps_decay = 1

    for ep in range(episodes):
        print(f"\nEvaluation Episode {ep+1}")
        try:
            state = env.reset()
            if state is None:
                print("Failed to reset environment, skipping episode")
                continue

            for _ in range(3):
                try:
                    traci.simulationStep()
                    print("Initial vehicle load check:", traci.simulation.getMinExpectedNumber())
                except Exception as e:
                    raise RuntimeError("TraCI lost during warmup") from e

            total_reward, step = 0, 0

            while True:
                try:
                    print(f"\rStep {step}/{env.max_steps}", end="")
                    action = agent.select_action(state)
                    next_state, reward, done = env.step(action)
                    state = next_state
                    total_reward += reward
                    step += 1
                    time.sleep(0.1)  # visualize in GUI
                    if done:
                        break
                except Exception as e:
                    raise RuntimeError(f"TraCI failed at step {step}") from e

            print(f"\nTotal Reward: {total_reward:.2f} | Steps: {step}")
            print_metrics(env)

        except Exception as e:
            print(f"\nError during episode: {e}")
            if hasattr(env, '_cleanup_connection'):
                env._cleanup_connection()
            time.sleep(2)
            continue

def print_metrics(env):
    """Helper to print evaluation metrics."""
    try:
        if env.episode_metrics['speeds']:
            avg_speed = sum(env.episode_metrics['speeds']) / len(env.episode_metrics['speeds'])
            print(f"Avg Speed: {avg_speed:.2f} mph")
        if env.episode_metrics['waiting_times']:
            print(f"Avg Waiting Time: {sum(env.episode_metrics['waiting_times']) / len(env.episode_metrics['waiting_times']):.2f} s")
        if env.episode_metrics['queue_lengths']:
            print(f"Avg Queue Length: {sum(env.episode_metrics['queue_lengths']) / len(env.episode_metrics['queue_lengths']):.2f}")
        if env.episode_metrics['travel_times']:
            print(f"Avg Travel Time: {sum(env.episode_metrics['travel_times']) / len(env.episode_metrics['travel_times']):.2f} s")
        # Print average stopped vehicles if available
        if 'traffic_stats' in env.episode_metrics and env.episode_metrics['traffic_stats']:
            stopped = [s.get('stopped_vehicles', 0) for s in env.episode_metrics['traffic_stats'] if s]
            if stopped:
                avg_stopped = sum(stopped) / len(stopped)
                print(f"Avg Stopped Vehicles: {avg_stopped:.2f}")
    except Exception as e:
        print("Metric reporting error:", e)

if __name__ == "__main__":
    print("Loading agent and environment...")

    try:
        sumocfg_path = os.path.join(os.path.dirname(__file__), "osm.sumocfg")

        # Clean up any running SUMO instances
        if os.name == 'nt':
            os.system('taskkill /f /im sumo-gui.exe >nul 2>&1')
            os.system('taskkill /f /im sumo.exe >nul 2>&1')
        else:
            os.system('pkill -f sumo-gui')
            os.system('pkill -f sumo')
        time.sleep(2)

        env = TrafficEnvironment(sumocfg_file=sumocfg_path, use_gui=True, max_steps=3600)
        print("SUMO GUI launched successfully.")

        agent = DQNAgent(state_size=49, junction_actions=env.junction_actions, device=device)
        print("Agent initialized.")

        print("Loading trained weights...")
        agent.load("best_model.pth")
        print("Model loaded.")

        try:
            traci.close()
        except Exception:
            pass

        evaluate(agent, env)
        print("Evaluation complete.")

    except Exception as e:
        print(f"\nFatal error: {e}")
    finally:
        print("Shutting down environment...")
        try:
            env.close()
        except Exception as e:
            print(f"Error while closing env: {e}")
        print("Done.")
