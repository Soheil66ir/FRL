import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    """
    Actor network for TD3 agent.
    """
    def __init__(self, state_dim, goal_dim, action_dim, max_action):
        """
        Initialise the Actor network.

        Parameters:
         state_dim: Dimension of the state space.
         goal_dim: Dimension of the goal space.
         action_dim: Dimension of the action space.
         max_action: Maximum action value.
        """
        super(Actor, self).__init__()
        self.max_action = max_action
        input_dim = state_dim + goal_dim  # Concatenate state and goal dimensions
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.action_out = nn.Linear(256, action_dim)

    def forward(self, state, goal):
        """
        Forward pass to compute the action.

        Parameters:
         state: Current state.
         goal: Current goal.

        Returns:
         Action.
        """
        x = torch.cat([state, goal], dim=1)  # Concatenate state and goal
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        actions = self.max_action * torch.tanh(self.action_out(x))
        return actions

class Critic(nn.Module):
    """
    Critic network for TD3 agent.
    """
    def __init__(self, state_dim, goal_dim, action_dim):
        """
        Initialise the Critic network.

        Parameters:
         state_dim: Dimension of the state space.
         goal_dim: Dimension of the goal space.
         action_dim: Dimension of the action space.
        """
        super(Critic, self).__init__()
        input_dim = state_dim + goal_dim + action_dim
        
        # Critic 1
        self.fc1_1 = nn.Linear(input_dim, 256)
        self.fc1_2 = nn.Linear(256, 256)
        self.fc1_3 = nn.Linear(256, 256)
        self.q1_out = nn.Linear(256, 1)
        
        # Critic 2
        self.fc2_1 = nn.Linear(input_dim, 256)
        self.fc2_2 = nn.Linear(256, 256)
        self.fc2_3 = nn.Linear(256, 256)
        self.q2_out = nn.Linear(256, 1)

    def forward(self, state, goal, action):
        """
        Forward pass to compute the Q-values.

        Parameters:
         state: Current state.
         goal: Current goal.
         action: Current action.

        Returns:
         Q-values (Q1, Q2).
        """
        x = torch.cat([state, goal, action], dim=1)
        
        # Q1
        q1 = F.relu(self.fc1_1(x))
        q1 = F.relu(self.fc1_2(q1))
        q1 = F.relu(self.fc1_3(q1))
        q1 = self.q1_out(q1)
        
        # Q2
        q2 = F.relu(self.fc2_1(x))
        q2 = F.relu(self.fc2_2(q2))
        q2 = F.relu(self.fc2_3(q2))
        q2 = self.q2_out(q2)
        
        return q1, q2
