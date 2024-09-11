import os
import gymnasium as gym
import numpy as np
import torch
import matplotlib.pyplot as plt
from Memory import HindsightReplayBuffer
from Agent import TD3
import pandas as pd
import flwr as fl
import time

# Initialise the gym environment and set the device (GPU if available, else CPU)
env_name = 'FetchPickAndPlace-v2'
env = gym.make(env_name)  # FetchPickAndPlace environment with sparse rewards
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Get state and action dimensions from the environment for the agent's networkss
state_dim = env.observation_space['observation'].shape[0]
goal_dim = env.observation_space['desired_goal'].shape[0]
achieved_goal_dim = env.observation_space['achieved_goal'].shape[0]
action_dim = env.action_space.shape[0]
max_action = env.action_space.high[0]

# Hyperparameters for the TD3 agent
actor_lr = 3e-4  # Learning rate for the actor network
critic_lr = 3e-4  # Learning rate for the critic network
discount = 0.98  # Discount factor for future rewards
tau = 0.005  # Soft update rate for target networks
batch_size = 512  # Batch size for training
epochs = 2  # Number of epochs for training
episodes_per_epoch = 200  # Number of episodes per epoch
max_timesteps = 50  # Max steps per episode
policy_noise = 0.2  # Noise added to policy actions
noise_clip = 0.5  # Maximum noise allowed during action selection
policy_freq = 2  # Frequency of policy updates

# Initialise the TD3 agent and hindsight experience replay buffer
agent = TD3(state_dim, goal_dim, action_dim, max_action, device, discount, tau, actor_lr, critic_lr, policy_noise, noise_clip, policy_freq)
replay_buffer = HindsightReplayBuffer(env, state_dim, action_dim, goal_dim, device, max_size=int(1e6))

# Create a directory to save models and training metrics
metrics_dir = os.path.join(os.path.dirname(__file__), "metrics")
if not os.path.exists(metrics_dir):
    os.makedirs(metrics_dir)

# Define a custom Flower Client class to interact with the federated learning server
class Client(fl.client.NumPyClient):
    def __init__(self, agent, replay_buffer, env, epochs, episodes_per_epoch, batch_size, max_timesteps, metrics_dir, client_id):
        self.agent = agent  # TD3 agent
        self.replay_buffer = replay_buffer  # Replay buffer for experience
        self.env = env  # Gym environment
        self.epochs = epochs  # Number of epochs
        self.episodes_per_epoch = episodes_per_epoch  # Episodes per epoch
        self.batch_size = batch_size  # Batch size for training
        self.max_timesteps = max_timesteps  # Maximum steps per episode
        self.metrics_dir = metrics_dir  # Directory to save metrics
        self.client_id = client_id  # Unique identifier for the client

        # Initialise lists to store metrics such as rewards, losses, success rates, and Q-values
        self.episode_rewards = []
        self.success_rates = []
        self.actor_losses = []
        self.critic_losses = []
        self.q1_values = []
        self.q2_values = []

        # Save hyperparameters to a CSV file
        hyperparameters = {
            'actor_lr': actor_lr,
            'critic_lr': critic_lr,
            'discount': discount,
            'tau': tau,
            'batch_size': batch_size,
            'epochs': epochs,
            'episodes_per_epoch': episodes_per_epoch,
            'max_timesteps': max_timesteps,
            'policy_noise': policy_noise,
            'noise_clip': noise_clip,
            'policy_freq': policy_freq
        }
        hyperparameters_df = pd.DataFrame([hyperparameters])
        hyperparameters_csv_path = os.path.join(metrics_dir, f'hyperparameters_{env_name}_{client_id}.csv')
        hyperparameters_df.to_csv(hyperparameters_csv_path, index=False)
        
    # Return current network parameters (actor and critic) as numpy arrays to server        
    def get_parameters(self, config):
        actor_params = [val.cpu().numpy() for val in self.agent.actor.state_dict().values()]
        critic_params = [val.cpu().numpy() for val in self.agent.critic.state_dict().values()]
        return actor_params + critic_params

    # Set the network parameters received from the server
    def set_parameters(self, parameters):
        actor_param_len = len(list(self.agent.actor.state_dict().keys()))
        actor_params = parameters[:actor_param_len]
        critic_params = parameters[actor_param_len:]

        actor_keys = list(self.agent.actor.state_dict().keys())
        critic_keys = list(self.agent.critic.state_dict().keys())

        actor_param_dict = {k: torch.tensor(v) for k, v in zip(actor_keys, actor_params)}
        critic_param_dict = {k: torch.tensor(v) for k, v in zip(critic_keys, critic_params)}

        self.agent.actor.load_state_dict(actor_param_dict)
        self.agent.critic.load_state_dict(critic_param_dict)

        # Update target networks for soft update after the global model aggregation
        self.agent.update_target_networks()

    # Perform training on the client, update local models, and collect metrics
    def fit(self, parameters, config):
        self.set_parameters(parameters) # Set parameters from the server
        episode_rewards = []
        success_rates = []
        actor_losses = []
        critic_losses = []
        q1_values = []
        q2_values = []
        
        # Run training for the specified number of epochs
        for epoch in range(self.epochs):
            epoch_rewards = []
            successes = 0
            for episode in range(self.episodes_per_epoch):
                obs = self.env.reset()[0]
                state = obs['observation']
                goal = obs['desired_goal']
                achieved_goal = obs['achieved_goal']

                episode_reward = 0
                episode_length = 0

                for t in range(self.max_timesteps):
                    episode_length += 1
                    action = self.agent.select_action(state, goal) # Select action
                    noise = np.random.normal(0, max_action * 0.1, size=action_dim) # Add noise to action for exploration
                    action = np.clip(action + noise, -max_action, max_action) # Clip action within bounds

                    # Step in the environment and collect new state, reward, and done flag  
                    next_obs, reward, done, truncated, info = self.env.step(action)[:5]
                    next_state = next_obs['observation']
                    next_goal = next_obs['desired_goal']
                    next_achieved_goal = next_obs['achieved_goal']

                    # If episode succeeded, mark as done
                    if info['is_success'] == 1.0:
                        successes += 1
                        done = True

                    # Add experience to the replay buffer
                    self.replay_buffer.add(state, action, next_state, reward, done, goal, achieved_goal, truncated, episode_length)
                    state = next_state
                    goal = next_goal
                    achieved_goal = next_achieved_goal
                    episode_reward += reward
                    if done:
                        break

                episode_rewards.append(episode_reward)
                epoch_rewards.append(episode_reward)
                
                # Train the agent if enough samples are available in the replay buffer
                if self.replay_buffer.size > self.batch_size:
                    actor_loss, critic_loss, q1_value, q2_value = self.agent.train(self.replay_buffer, self.batch_size)
                    actor_losses.append(actor_loss)
                    critic_losses.append(critic_loss)
                    q1_values.append(q1_value)
                    q2_values.append(q2_value)
                else:
                    actor_losses.append(None)
                    critic_losses.append(None)
                    q1_values.append(None)
                    q2_values.append(None)

            success_rate = successes / self.episodes_per_epoch
            success_rates.append(success_rate)
            average_reward = np.mean(epoch_rewards)
            print(f"Epoch: {epoch}, Success Rate: {success_rate}, Average Reward: {average_reward}")

        # Extend the lists with results from the current training 
        self.episode_rewards.extend(episode_rewards)
        self.success_rates.extend(success_rates)
        self.actor_losses.extend(actor_losses)
        self.critic_losses.extend(critic_losses)
        self.q1_values.extend(q1_values)
        self.q2_values.extend(q2_values)

        # Make sure all lists have the same length
        max_len = max(len(self.episode_rewards), len(self.success_rates), len(self.actor_losses), len(self.critic_losses), len(self.q1_values), len(self.q2_values))
        for metric in [self.episode_rewards, self.success_rates, self.actor_losses, self.critic_losses, self.q1_values, self.q2_values]:
            while len(metric) < max_len:
                metric.append(None)

        # Save metrics to a CSV file
        self.save_metrics()

        # Save the models
        self.save_models()

        # Return updated parameters and metrics
        metrics = {
            "success_rate": np.mean(success_rates),
            "average_reward": np.mean(episode_rewards),
        }

        return self.get_parameters(config), len(epoch_rewards), metrics
    
    # Evaluate the model using a number of episodes and return the average reward and success rate
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        
        # Set a default value for num_episodes if it's not in config
        num_episodes = config.get("num_episodes", 100)  # Default to 100 episodes if not specified
        
        # Evaluate the model in the FetchSlide environment
        total_reward = 0.0
        successes = 0
        
        for _ in range(num_episodes):
            obs = self.env.reset()[0]
            state = obs['observation']
            goal = obs['desired_goal']
            achieved_goal = obs['achieved_goal']

            episode_reward = 0
            for t in range(self.max_timesteps):
                action = self.agent.select_action(state, goal)
                next_obs, reward, done, truncated, info = self.env.step(action)[:5]
                state = next_obs['observation']
                goal = next_obs['desired_goal']
                achieved_goal = next_obs['achieved_goal']
                episode_reward += reward
                if info['is_success'] == 1.0:
                    successes += 1
                    done = True
                if done:
                    break
            total_reward += episode_reward
        
        average_reward = total_reward / num_episodes
        success_rate = successes / num_episodes
        metrics = {"success_rate": success_rate, "average_reward": average_reward}
        return float(average_reward), num_episodes, metrics
    
    # Additional training for the agent after federated learning rounds when the agents are performing different tasks
    def additional_training(self, additional_epochs, additional_episodes_per_epoch):
        self.epochs = additional_epochs
        self.episodes_per_epoch = additional_episodes_per_epoch
        self.fit(self.get_parameters({}), {})

    # Save metrics to a CSV file
    def save_metrics(self):
        metrics = {
            "episode_rewards": self.episode_rewards,
            "success_rates": self.success_rates,
            "actor_losses": self.actor_losses,
            "critic_losses": self.critic_losses,
            "q1_values": self.q1_values,
            "q2_values": self.q2_values,
        }
        metrics_df = pd.DataFrame(metrics)
        metrics_file = os.path.join(self.metrics_dir, f"metrics_{env_name}_{self.client_id}.csv")
        metrics_df.to_csv(metrics_file, index=False)

    # Save the models (actor and critic) to disk
    def save_models(self):
        actor_model_file = os.path.join(self.metrics_dir, f"actor_{env_name}_{self.client_id}.pth")
        critic_model_file = os.path.join(self.metrics_dir, f"critic_{env_name}_{self.client_id}.pth")
        torch.save(self.agent.actor.state_dict(), actor_model_file)
        torch.save(self.agent.critic.state_dict(), critic_model_file)

# Generate a unique identifier for the client
client_id = str(int(time.time()))

# Create and start the Flower client
if __name__ == "__main__":
    client = Client(agent, replay_buffer, env, epochs, episodes_per_epoch, batch_size, max_timesteps, metrics_dir, client_id)
    fl.client.start_client(server_address="localhost:8080", client=client.to_client())

    # Perform additional training after the federated learning rounds
    additional_epochs = 1
    additional_episodes_per_epoch = 200
    client.additional_training(additional_epochs, additional_episodes_per_epoch)
