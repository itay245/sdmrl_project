import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np

# Define the Actor-Critic Network
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(ActorCritic, self).__init__()
        
        # Actor network
        self.actor_fc1 = nn.Linear(state_dim, hidden_dim)
        self.actor_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_std = nn.Linear(hidden_dim, action_dim)
        
        # Critic network
        self.critic_fc1 = nn.Linear(state_dim, hidden_dim)
        self.critic_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.critic_value = nn.Linear(hidden_dim, 1)
        
    def forward(self, state):
        # Actor forward pass
        x = torch.relu(self.actor_fc1(state))
        x = torch.relu(self.actor_fc2(x))
        mean = torch.tanh(self.actor_mean(x))  # Tanh to bound output
        std = nn.functional.softplus(self.actor_std(x)) + 1e-6  # Ensure positive std
        
        # Critic forward pass
        v = torch.relu(self.critic_fc1(state))
        v = torch.relu(self.critic_fc2(v))
        value = self.critic_value(v)
        
        return mean, std, value

# PPO Agent Class
class PPO_Agent:
    def __init__(self, env, learning_rate=0.0003, gamma=0.99, eps_clip=0.2, update_steps=5,weighy_path="PPO_policy.pth",device="cpu"):
        self.env = env
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.update_steps = update_steps
        self.policy = ActorCritic(state_dim=3, action_dim=1).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.weight_path = weighy_path
        self.device = device      
    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        mean, std, _ = self.policy(state)
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob
    
    def compute_advantages(self, rewards, values, dones):
        advantages = []
        returns = []
        G = 0
        for i in reversed(range(len(rewards))):
            G = rewards[i] + self.gamma * G * (1 - dones[i])
            returns.insert(0, G)
            advantages.insert(0, G - values[i])
        return torch.tensor(advantages, dtype=torch.float32).to(self.device), torch.tensor(returns, dtype=torch.float32).to(self.device)
    
    def train(self, num_episodes=1000, batch_size=32):
        total_rewards = []
        for episode in range(num_episodes):
            state = self.env.reset()
            log_probs = []
            values = []
            rewards = []
            dones = []
            states = []
            actions = []
            done = False
            
            while not done:
                action, log_prob = self.select_action(state)
                next_state, reward, done, _ = self.env.step([action])
                _, _, value = self.policy(torch.tensor(state, dtype=torch.float32).to(self.device))
                
                states.append(state)
                actions.append(action)
                log_probs.append(log_prob)
                values.append(value.item())
                rewards.append(reward)
                dones.append(done)
                
                state = next_state
            
            advantages, returns = self.compute_advantages(rewards, values, dones)
            
            # Policy and Value Function Optimization
            for _ in range(self.update_steps):
                states_tensor = torch.tensor(states, dtype=torch.float32).to(self.device)
                actions_tensor = torch.tensor(actions, dtype=torch.float32).to(self.device)
                old_log_probs_tensor = torch.stack(log_probs).detach()
                
                mean, std, value_preds = self.policy(states_tensor)
                dist = Normal(mean, std)
                new_log_probs = dist.log_prob(actions_tensor)
                
                ratio = torch.exp(new_log_probs - old_log_probs_tensor)
                clipped_ratio = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip)
                policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
                
                value_loss = nn.MSELoss()(value_preds.squeeze(), returns)
                
                loss = policy_loss + 0.5 * value_loss
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if episode % 10 == 0:
                print(f"Episode {episode}, Total Reward: {sum(rewards):.2f}")
            total_rewards.append(sum(rewards))
        torch.save(self.policy.state_dict(), self.weight_path)
        print(f"Model saved as {self.weight_path}")
        return total_rewards
    def evaluate(self, num_episodes=100):
        total_rewards = []
        for episode in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action, _ = self.select_action(state)
                next_state, reward, done, _ = self.env.step([action])
                episode_reward += reward
                state = next_state
            
            total_rewards.append(episode_reward)
        
        avg_reward = np.mean(total_rewards)
        std_reward = np.std(total_rewards)
        print(f"Evaluation over {num_episodes} episodes: Avg Reward = {avg_reward:.2f}, Std Dev = {std_reward:.2f}")
        return avg_reward, std_reward



