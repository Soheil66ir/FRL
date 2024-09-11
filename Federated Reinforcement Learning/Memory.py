import numpy as np
import torch
import random

class ReplayBuffer:
    """
    Replay buffer for storing transitions
    """
    def __init__(self, state_dim, action_dim, goal_dim, device, max_size=int(1e6)):
        """
        Initialise the replay buffer.

        Parameters:
         state_dim: Dimension of the state space
         action_dim: Dimension of the action space.
         goal_dim: Dimension of the goal space
         device: Device to store the tensors on
         max_size: Maximum size of the buffer
        """
        self.device = device
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        
        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.done = np.zeros((max_size, 1))
        self.goal = np.zeros((max_size, goal_dim))
        self.achieved_goal = np.zeros((max_size, goal_dim))
        
    def add(self, state, action, next_state, reward, done, goal, achieved_goal):
        """
        Add a new transition to the buffer 

        Parameters:
         state: Current state
         action: Action taken based on state
         next_state: Next state
         reward: Reward
         done: Whether the episode is done
         goal: Goal
         achieved_goal: Achieved goal
        """
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.done[self.ptr] = done
        self.goal[self.ptr] = goal
        self.achieved_goal[self.ptr] = achieved_goal
        
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        """
        Sample a batch of transitions from the buffer

        Parameters:
         batch_size: Size of the batch for sample

        Returns:
        A tuble of batch of transitions (state, action, next_state, reward, done, goal, achieved_goal)
        """
        ind = np.random.randint(0, self.size, size=batch_size)
        state = torch.FloatTensor(self.state[ind]).to(self.device)
        action = torch.FloatTensor(self.action[ind]).to(self.device)
        next_state = torch.FloatTensor(self.next_state[ind]).to(self.device)
        reward = torch.FloatTensor(self.reward[ind]).to(self.device)
        done = torch.FloatTensor(1 - self.done[ind]).to(self.device)  # Convert done to not_done
        goal = torch.FloatTensor(self.goal[ind]).to(self.device)
        achieved_goal = torch.FloatTensor(self.achieved_goal[ind]).to(self.device)
        
        return state, action, next_state, reward, done, goal, achieved_goal

class HindsightReplayBuffer(ReplayBuffer):
    """
    Hindsight Experience Replay (HER) buffer for storing and sampling experiences with future goals according to HER paper
    """
    def __init__(self, env, state_dim, action_dim, goal_dim, device, max_size=int(1e6), her_k=4):
        """
        Initialise the HER replay buffer

        Parameters:
         env: Environmet name
         state_dim: Dimension of the state space
         action_dim: Dimension of the action space
         goal_dim: Dimension of the goal space
         device: Device to store the tensors on
         max_size: Maximum size of the buffer.
         her_k: Number of HER transitions to add for each episode
        """
        super(HindsightReplayBuffer, self).__init__(state_dim, action_dim, goal_dim, device, max_size)
        self.env = env
        self.her_k = her_k
        self.episode_storage = []

    def add(self, state, action, next_state, reward, done, goal, achieved_goal, truncated, episode_length):
        """
        Add transitions and HER transitions to the buffer.

        Parameters:
         state: Current state
         action: Action taken based on state
         next_state: Next state
         reward: Reward
         done: Whether the episode is done
         goal: Goal
         achieved_goal: Achieved goal
         truncated: Whether the episode is truncated.
         episode_length: Length of the episode to collect HER sample(according to Future HER from the paper, which collect samples from latest episodes)
        """
        super().add(state, action, next_state, reward, done, goal, achieved_goal)
        self.episode_storage.append((state, action, next_state, reward, done, goal, achieved_goal))

        if truncated or done:
            episode = self.episode_storage[-episode_length:]
            episode_indices = list(range(episode_length))
            selected_indices = np.random.choice(episode_indices, min(self.her_k, episode_length), replace=False)
            
            for index in selected_indices:
                future = np.random.randint(index, episode_length)
                future_goal = episode[future][6]  # Extract achieved_goal from future transition

                # Create HER transition
                state, action, next_state, _, _, goal, achieved_goal = episode[index]
                her_done = True  # HER transitions are treated as done
                her_reward = self.env.unwrapped.compute_reward(future_goal, future_goal, None)
                
                # Adding HER transition
                super().add(state, action, next_state, her_reward, her_done, future_goal, future_goal)
            
            self.episode_storage.clear()  # Clear the episode storage after processing

    def add(self, state, action, next_state, reward, done, goal, achieved_goal, truncated, episode_length):
        super().add(state, action, next_state, reward, done, goal, achieved_goal)
        self.episode_storage.append((state, action, next_state, reward, done, goal, achieved_goal))

        if truncated or done:
            episode = self.episode_storage[-episode_length:]
            episode_indices = list(range(episode_length))
            selected_indices = np.random.choice(episode_indices, min(self.her_k, episode_length), replace=False)
            
            for index in selected_indices:
                future = np.random.randint(index, episode_length)
                future_goal = episode[future][6]  # Extract achieved_goal from future transition

                # Create HER transition
                state, action, next_state, _, _, goal, achieved_goal = episode[index]
                her_done = True  # Usually, HER transitions are treated as done
                her_reward = self.env.unwrapped.compute_reward(future_goal, future_goal, None)
                
                # Adding HER transition
                super().add(state, action, next_state, her_reward, her_done, future_goal, future_goal)
            
            self.episode_storage.clear()  # Clear the episode storage after processing


