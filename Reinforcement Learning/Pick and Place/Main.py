import os
import gymnasium as gym
import numpy as np
import torch
import matplotlib.pyplot as plt
from Memory import HindsightReplayBuffer
from Agent import TD3
import pandas as pd

# Initialise environment and device
env_name = 'FetchPickAndPlace-v2'
env = gym.make(env_name)  # Using sparse reward
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Get state and action dimensions for Networks
state_dim = env.observation_space['observation'].shape[0]
goal_dim = env.observation_space['desired_goal'].shape[0]
achieved_goal_dim = env.observation_space['achieved_goal'].shape[0]
action_dim = env.action_space.shape[0]
max_action = env.action_space.high[0]

# Hyperparameters
actor_lr = 3e-4
critic_lr = 3e-4
discount = 0.98
tau = 0.005
batch_size = 512
epochs = 1000
episodes_per_epoch = 200
max_timesteps = 50
policy_noise = 0.2
noise_clip = 0.5
policy_freq = 2

# Initialise agent and HER replay buffer
agent = TD3(state_dim, goal_dim, action_dim, max_action, device, discount, tau, actor_lr, critic_lr, policy_noise, noise_clip, policy_freq)
replay_buffer = HindsightReplayBuffer(env, state_dim, action_dim, goal_dim, device, max_size=int(1e6))
obs_shape = state_dim + goal_dim + achieved_goal_dim # This was for normalisation for DDPG, however TD3 does not need that.

# Directory to save models and graphs
save_dir = os.path.join(os.path.dirname(__file__), f"models")
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Paths to the save Network's model
actor_path = os.path.join(save_dir, "final_actor.pth")
critic_path = os.path.join(save_dir, "final_critic.pth")
actor_target_path = os.path.join(save_dir, "final_actor_target.pth")
critic_target_path = os.path.join(save_dir, "final_critic_target.pth")

# Load the saved Network's model if exist
if os.path.exists(actor_path):
    agent.actor.load_state_dict(torch.load(actor_path))
    agent.critic.load_state_dict(torch.load(critic_path))
    agent.actor_target.load_state_dict(torch.load(actor_target_path))
    agent.critic_target.load_state_dict(torch.load(critic_target_path))
    print("Loaded saved models for actor and critic.")
else:
    print("No saved models found. Starting training from scratch.")

# Training loop
# List to store Metrics during training
episode_rewards = []
success_rates = []
actor_losses = []
critic_losses = []
q1_values = []
q2_values = []

for epoch in range(epochs):
    # List to store rewards for each episode in the current epoch
    epoch_rewards = []
    # Counter to keep track of successful episodes in every epoch
    successes = 0
    
    for episode in range(episodes_per_epoch):
        # Reset the environment to get the initial observation
        obs = env.reset()[0]
        state = obs['observation']
        goal = obs['desired_goal']
        achieved_goal = obs['achieved_goal']

        # Initialise variables to store the reward and length of the episode for HER
        episode_reward = 0
        episode_length = 0

        for t in range(max_timesteps): # Defualt max_timesteps is 50  in robatic envs
            episode_length += 1
            # Select an action based on the current policy
            action = agent.select_action(state, goal)
            # Add noise to the action for exploration
            noise = np.random.normal(0, max_action * 0.1, size=action_dim)
            action = np.clip(action + noise, -max_action, max_action)

            # Take a step in the environment with the action
            next_obs, reward, done, truncated, info = env.step(action)[:5]
            next_state = next_obs['observation']
            next_goal = next_obs['desired_goal']
            next_achieved_goal = next_obs['achieved_goal']

            # Check if the episode was successful to add a success to counter and change done flag
            if info['is_success'] == 1.0:
                successes += 1
                done = True

            # Add the transition to the replay buffer
            replay_buffer.add(state, action, next_state, reward, done, goal, achieved_goal, truncated, episode_length)
            
            # Update the state, goal, and achieved goal for the next timestep
            state = next_state
            goal = next_goal
            achieved_goal = next_achieved_goal
            
            # Accumulate the reward for the current episode
            episode_reward += reward

            # If the episode is done, exit the loop. since the loop is for max_timesteps, if the agent does not achive the goal, in final step, truncated will change to True. if any of done or truncated will be true, it will activate HER loop in Memory
            if done:
                break
        
        # Store the cumulative reward for the episode and epoch
        episode_rewards.append(episode_reward)
        epoch_rewards.append(episode_reward)
        
        # Training the agent 
        if replay_buffer.size > batch_size:
            actor_loss, critic_loss, q1_value, q2_value = agent.train(replay_buffer, batch_size)
            actor_losses.append(actor_loss)
            critic_losses.append(critic_loss)
            q1_values.append(q1_value)
            q2_values.append(q2_value)
        else:
            actor_losses.append(None)
            critic_losses.append(None)
            q1_values.append(None)
            q2_values.append(None)
    
    # Calculating the metrics for epoch
    success_rate = successes / episodes_per_epoch
    success_rates.append(success_rate)
    average_reward = np.mean(epoch_rewards)
    print(f"Epoch: {epoch}, Success Rate: {success_rate}, Average Reward: {average_reward}")

# Save the final Network's model
final_actor_path = os.path.join(save_dir, "final_actor.pth")
final_critic_path = os.path.join(save_dir, "final_critic.pth")
final_actor_target_path = os.path.join(save_dir, "final_actor_target.pth")
final_critic_target_path = os.path.join(save_dir, "final_critic_target.pth")

torch.save(agent.actor.state_dict(), final_actor_path)
torch.save(agent.critic.state_dict(), final_critic_path)
torch.save(agent.actor_target.state_dict(), final_actor_target_path)
torch.save(agent.critic_target.state_dict(), final_critic_target_path)

# Pad arrays to ensure they are of the same length for savinf in CSV file
max_length = max(len(episode_rewards), len(success_rates), len(actor_losses), len(critic_losses), len(q1_values), len(q2_values))
episode_rewards.extend([None] * (max_length - len(episode_rewards)))
success_rates.extend([None] * (max_length - len(success_rates)))
actor_losses.extend([None] * (max_length - len(actor_losses)))
critic_losses.extend([None] * (max_length - len(critic_losses)))
q1_values.extend([None] * (max_length - len(q1_values)))
q2_values.extend([None] * (max_length - len(q2_values)))

# Save metrics to CSV file
metrics_df = pd.DataFrame({
    'episode_rewards': episode_rewards,
    'success_rates': success_rates,
    'actor_losses': actor_losses,
    'critic_losses': critic_losses,
    'q1_values': q1_values,
    'q2_values': q2_values
})
metrics_csv_path = os.path.join(save_dir, 'metrics.csv')
metrics_df.to_csv(metrics_csv_path, index=False)

# Save hyperparameters to CSV file
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
hyperparameters_csv_path = os.path.join(save_dir, 'hyperparameters.csv')
hyperparameters_df.to_csv(hyperparameters_csv_path, index=False)

# Metrics plot
plt.figure(figsize=(12, 6))
plt.plot(episode_rewards)
plt.xlabel('Episode')
plt.ylabel('Accumulated Reward')
plt.title('Accumulated Reward per Episode')
reward_plot_path = os.path.join(save_dir, 'accumulated_rewards.png')
plt.savefig(reward_plot_path)
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(success_rates)
plt.xlabel('Epoch')
plt.ylabel('Success Rate')
plt.title('Success Rate per Epoch')
success_rate_plot_path = os.path.join(save_dir, 'success_rates.png')
plt.savefig(success_rate_plot_path)
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(actor_losses)
plt.xlabel('Training Step')
plt.ylabel('Actor Loss')
plt.title('Actor Loss during Training')
actor_loss_plot_path = os.path.join(save_dir, 'actor_losses.png')
plt.savefig(actor_loss_plot_path)
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(critic_losses)
plt.xlabel('Training Step')
plt.ylabel('Critic Loss')
plt.title('Critic Loss during Training')
critic_loss_plot_path = os.path.join(save_dir, 'critic_losses.png')
plt.savefig(critic_loss_plot_path)
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(q1_values, label='Q1 Values', color='blue')
plt.plot(q2_values, label='Q2 Values', color='red')
plt.xlabel('Training Step')
plt.ylabel('Q Value')
plt.title('Q Values during Training')
plt.legend()
q_value_plot_path = os.path.join(save_dir, 'q_values_combined.png')
plt.savefig(q_value_plot_path)
plt.show()

# Rendering the saved model
def render_model(agent, actor_model_path, num_episodes=20, max_timesteps=50):
    """
    Render the trained model in the environment.

    Parameters:
     agent: Trained TD3 agent.
     actor_model_path: Path to the saved actor model.
     num_episodes: Number of episodes to render.
     max_timesteps: Maximum number of timesteps per episode.
    """
    env_render = gym.make(env_name, render_mode='human')  
    agent.actor.load_state_dict(torch.load(actor_model_path))
    agent.actor.eval()

    for episode in range(num_episodes):
        obs = env_render.reset()[0]
        state = obs['observation']
        goal = obs['desired_goal']
        achieved_goal = obs['achieved_goal']

        for t in range(max_timesteps):
            action = agent.select_action(state, goal)
            next_obs, reward, done, info = env_render.step(action)[:4]
            next_state = next_obs['observation']
            next_goal = next_obs['desired_goal']
            next_achieved_goal = next_obs['achieved_goal']

            state = next_state
            goal = next_goal
            achieved_goal = next_achieved_goal

            if reward == 0:
                done = True
            env_render.render()
            if done:
                break
    env_render.close()

# Use the final saved model for rendering
render_model(agent, final_actor_path, num_episodes=20, max_timesteps=50)

# Close the training environment
env.close()