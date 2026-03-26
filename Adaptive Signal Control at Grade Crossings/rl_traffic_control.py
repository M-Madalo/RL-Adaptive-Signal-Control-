import os
import torch
import numpy as np
from traffic_components import device, performance_targets
from reinforcement_learningCA import TrafficEnvironment, DQNAgent

def test_saved_model(model_path, num_episodes=5):
    """
    Test a saved model and return performance metrics.
    
    Args:
        model_path (str): Path to the saved model file
        num_episodes (int): Number of episodes to run for testing
        
    Returns:
        dict: Dictionary containing performance metrics
    """
    # Initialize environment
    env = TrafficEnvironment(os.path.join(os.path.dirname(__file__), "osm.sumocfg"))
    
    # Initialize agent
    agent = DQNAgent(
        state_size=49,
        junction_actions=env.junction_actions,
        device=device
    )
    
    # Load the saved model
    try:
        checkpoint = torch.load(model_path, map_location=device)
        agent.policy_net.load_state_dict(checkpoint['model_state_dict'])
        print(f"Successfully loaded model from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    
    # Set agent to evaluation mode
    agent.policy_net.eval()
    
    # Initialize metrics storage
    metrics = {
        'waiting_times': [],
        'speeds': [],
        'queues': [],
        'travel_times': [],
        'emergency_braking': [],
        'rewards': []
    }
    
    print("\nStarting model evaluation...")
    
    # Run test episodes
    for episode in range(num_episodes):
        print(f"\nTest Episode {episode + 1}/{num_episodes}")
        state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # Select action using the trained policy
            with torch.no_grad():
                action = agent.select_action(state)
            
            # Take action in environment
            next_state, reward, done = env.step(action)
            state = next_state
            episode_reward += reward
        
        # Store episode metrics
        metrics['waiting_times'].append(np.mean(env.episode_metrics['waiting_times']))
        metrics['speeds'].append(np.mean(env.episode_metrics['speeds']))
        metrics['queues'].append(np.mean(env.episode_metrics['queue_lengths']))
        metrics['travel_times'].append(np.mean(env.episode_metrics['travel_times']))
        metrics['emergency_braking'].append(env.emergency_braking_events)
        metrics['rewards'].append(episode_reward)
        
        # Print episode summary
        print(f"Episode {episode + 1} Summary:")
        print(f"Reward: {episode_reward:.2f}")
        print(f"Average Waiting Time: {metrics['waiting_times'][-1]:.2f}s")
        print(f"Average Speed: {metrics['speeds'][-1]:.2f} mph")
        print(f"Average Queue Length: {metrics['queues'][-1]:.2f}")
        print(f"Average Travel Time: {metrics['travel_times'][-1]:.2f}s")
        print(f"Emergency Braking Events: {metrics['emergency_braking'][-1]}")
    
    # Close environment
    env.close()
    
    # Calculate and print average metrics
    print("\nOverall Performance Metrics:")
    print(f"Average Reward: {np.mean(metrics['rewards']):.2f}")
    print(f"Average Waiting Time: {np.mean(metrics['waiting_times']):.2f}s")
    print(f"Average Speed: {np.mean(metrics['speeds']):.2f} mph")
    print(f"Average Queue Length: {np.mean(metrics['queues']):.2f}")
    print(f"Average Travel Time: {np.mean(metrics['travel_times']):.2f}s")
    print(f"Average Emergency Braking Events: {np.mean(metrics['emergency_braking']):.2f}")
    
    return metrics

if __name__ == "__main__":
    # Test the model if this file is run directly
    model_path = "best_model.pth"
    if os.path.exists(model_path):
        test_saved_model(model_path)
    else:
        print(f"Model file {model_path} not found. Please train the model first.") 