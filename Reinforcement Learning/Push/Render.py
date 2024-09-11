import gymnasium as gym
import torch
import os
import time  
from Agent import TD3

# Initialise environment and device
env = gym.make('FetchPush-v2', render_mode='human')  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Get state and action dimensions
state_dim = env.observation_space['observation'].shape[0]
goal_dim = env.observation_space['desired_goal'].shape[0]
achieved_goal_dim = env.observation_space['achieved_goal'].shape[0]
action_dim = env.action_space.shape[0]
max_action = env.action_space.high[0]

# Initialise agent, replay buffer, and noise
actor_lr = 3e-4
critic_lr = 3e-4
discount = 0.99
tau = 0.005
policy_noise = 0.0
noise_clip = 0.5
policy_freq = 2

# Initialise agent
agent = TD3(state_dim, goal_dim, action_dim, max_action, device, discount, tau, actor_lr, critic_lr, policy_noise, noise_clip, policy_freq)

# Define paths for loading models 
save_dir = os.path.join(os.path.dirname(__file__), "TrainedModels")
actor_model_path = os.path.join(save_dir, "final_actor.pth")

# Load the trained actor model
agent.actor.load_state_dict(torch.load(actor_model_path))
agent.actor.eval()

def render_model(agent, num_episodes, max_timesteps, delay=0.08):
    for episode in range(num_episodes):
        obs = env.reset()[0]
        state = obs['observation']
        goal = obs['desired_goal']
        achieved_goal = obs['achieved_goal']

        for t in range(max_timesteps):
            action = agent.select_action(state, goal)
            next_obs, reward, done, info = env.step(action)[:4]
            next_state = next_obs['observation']
            next_goal = next_obs['desired_goal']
            next_achieved_goal = next_obs['achieved_goal']

            state = next_state
            goal = next_goal
            achieved_goal = next_achieved_goal

            env.render()
            time.sleep(delay)  # Add delay to control FPS
            if reward == 0:
                done = True
                break
            

          
    env.close()

# Use the final saved model for rendering
render_model(agent, num_episodes=20, max_timesteps=50)
