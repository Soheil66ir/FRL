import torch
import torch.optim as optim
import torch.nn.functional as F
from Networks import Actor, Critic

class TD3:
    """
    TD3 (Twin Delayed DDPG) agent
    """
    def __init__(self, state_dim, goal_dim, action_dim, max_action, device, discount, tau, actor_lr, critic_lr, policy_noise, noise_clip, policy_freq):
        """
        Initialise the TD3 agent.

        Parameters:
         state_dim (int): Dimension of the state space
         goal_dim (int): Dimension of the goal space
         action_dim (int): Dimension of the action space
         max_action (float): Maximum action value
         device: Device to run the model on (CPU or GPU)
         discount (float): Discount factor for future rewards
         tau (float): Soft update parameter for target networks
         actor_lr (float): Learning rate for the actor network
         critic_lr (float): Learning rate for the critic network
         policy_noise (float): Noise added to target policy during critic update
         noise_clip (float): Range to clip noise
         policy_freq (int): Frequency of delayed policy updates
        """
        self.device = device

        # Initialise actor network and target actor network
        self.actor = Actor(state_dim, goal_dim, action_dim, max_action).to(self.device)
        self.actor_target = Actor(state_dim, goal_dim, action_dim, max_action).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)

        # Initialise critic networks and target critic networks
        self.critic = Critic(state_dim, goal_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, goal_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.total_it = 0

    def select_action(self, state, goal):
        """
        Select an action based on the current policy.

        Parameters:
         state: Current state.
         goal: Current goal.

        Returns:
         Selected action.
        """
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        goal = torch.FloatTensor(goal.reshape(1, -1)).to(self.device)
        return self.actor(state, goal).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size):
        """
        Train the agent using a batch of experiences from the HER replay buffer according to TD3 paper.

        Parameters:
         replay_buffer: Replay buffer for storing experiences.
         batch_size (int): Size of the batch for sample from the replay buffer.

        Returns:
         Actor loss.
         Critic loss.
         Q1 value.
         Q2 value.
        """
        self.total_it += 1

        # Sample a batch from the replay buffer
        state, action, next_state, reward, not_done, goal, achieved_goal = replay_buffer.sample(batch_size)

        # Compute the target Q value
        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state, goal) + noise).clamp(-self.max_action, self.max_action)

            target_Q1, target_Q2 = self.critic_target(next_state, goal, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + self.discount * target_Q * not_done

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, goal, action)

        # Compute critic losses
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimise the critics
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates according to TD3 algorithm
        actor_loss = 0
        if self.total_it % self.policy_freq == 0:
            # Compute actor loss using the minimum of the two Q-values
            actor_loss = -self.critic(state, goal, self.actor(state, goal))[0].mean()

            # Optimise the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return actor_loss.item() if self.total_it % self.policy_freq == 0 else 0, critic_loss.item(), current_Q1.mean().item(), current_Q2.mean().item()
