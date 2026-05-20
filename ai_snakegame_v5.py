import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
from collections import deque
from torch_geometric.data import Data
from queue import Queue
import copy

# Game configuration
WIDTH = 300
HEIGHT = 300
SPACE_SIZE = 20
BODY_SIZE = 2

# Enhanced RL parameters
EPISODES = 10000  # Increased training episodes
GAMMA = 0.997    # Slightly adjusted discount factor
LEARNING_RATE = 3e-5  # Further reduced for stability
EPSILON = 1.0
EPSILON_DECAY = 0.9997
EPSILON_MIN = 0.005
BATCH_SIZE = 1024
MEMORY_SIZE = 200000  # Doubled memory size
SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
SAVE_PATH = os.path.join(SAVE_DIR, "modelv5.pt")
UPDATE_TARGET_EVERY = 2
N_ATOMS = 51  # Number of atoms for distributional RL
V_MIN = -200  # Minimum value for distribution
V_MAX = 200   # Maximum value for distribution

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.017):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.out_features))

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)

class RelativePositionalEncoding(nn.Module):
    def __init__(self, dim, max_seq_len=100):
        super().__init__()
        pe = torch.zeros(max_seq_len, dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:seq_len, :]

class EnhancedTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        attended = self.attention(x, x, x)[0]
        x = self.norm1(x + self.dropout(attended))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

class DistributionalDQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.action_dim = action_dim
        self.n_atoms = N_ATOMS
        self.v_min = V_MIN
        self.v_max = V_MAX
        self.delta_z = (V_MAX - V_MIN) / (N_ATOMS - 1)
        
        # Fixed feature extraction with correct dimensions
        self.feature_net = nn.Sequential(
            NoisyLinear(state_dim, 256),  # Reduced initial dimension
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            NoisyLinear(256, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # Relative positional encoding
        self.pos_encoding = RelativePositionalEncoding(512)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            EnhancedTransformerBlock(512)
            for _ in range(4)
        ])
        
        # Value and Advantage streams for each atom
        self.value_net = nn.Sequential(
            NoisyLinear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            NoisyLinear(256, self.n_atoms)
        )
        
        self.advantage_net = nn.Sequential(
            NoisyLinear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            NoisyLinear(256, action_dim * self.n_atoms)
        )
        
        # Initialize support for value distribution
        self.register_buffer('supports', torch.linspace(V_MIN, V_MAX, N_ATOMS))
        
    def reset_noise(self):
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()
                
    def forward(self, state, return_distribution=False):
        batch_size = state.size(0) if len(state.size()) > 1 else 1
        if len(state.size()) == 1:
            state = state.unsqueeze(0)
            
        x = self.feature_net(state)
        x = x.unsqueeze(1)  # Add sequence dimension
        x = self.pos_encoding(x)
        
        for block in self.transformer_blocks:
            x = block(x)
            
        x = x.squeeze(1)  # Remove sequence dimension
        
        value = self.value_net(x).view(batch_size, 1, self.n_atoms)
        advantage = self.advantage_net(x).view(batch_size, self.action_dim, self.n_atoms)
        
        q_dist = value + advantage - advantage.mean(dim=1, keepdim=True)
        q_dist = F.softmax(q_dist, dim=-1)
        
        if return_distribution:
            return q_dist
            
        q_values = (q_dist * self.supports).sum(dim=-1)
        return q_values

# Add Monte Carlo Tree Search
class MCTSNode:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = {}
        self.visits = 0
        self.value = 0
        self.untried_actions = list(range(4))  # 4 possible actions
        
    def select_child(self, c_puct=1.0):
        return max(self.children.items(),
                  key=lambda item: item[1].value / (item[1].visits + 1e-8) + 
                  c_puct * math.sqrt(self.visits) / (1 + item[1].visits))
                  
    def expand(self, action, next_state):
        child = MCTSNode(next_state, self, action)
        self.children[action] = child
        return child
        
    def update(self, reward):
        self.visits += 1
        self.value += (reward - self.value) / self.visits

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = 0.001
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
        
    def push(self, experience):
        max_priority = self.priorities.max() if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.pos] = experience
            
        self.priorities[self.pos] = max_priority
        self.pos = (self.pos + 1) % self.capacity
        
    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return None
            
        probs = self.priorities[:len(self.buffer)] ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        return samples, indices, weights
        
    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

class EnhancedSnakeGameAI:
    def __init__(self):
        # Create save directory if it doesn't exist
        if not os.path.exists(SAVE_DIR):
            os.makedirs(SAVE_DIR)
            
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Game state initialization
        self.episode = 1
        self.total_steps = 0
        self.steps_without_food = 0
        self.score = 0
        self.direction = 'right'
        self.snake_coords = [[100, 100], [80, 100], [60, 100]]
        self.food_coord = self.place_food()
        
        # Initialize networks with correct state dimensions
        state_dim = 12  # Remove the +128 as we're not using that additional dimension
        action_dim = 4
        self.policy_net = DistributionalDQNetwork(state_dim, action_dim).to(self.device)
        self.target_net = DistributionalDQNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.memory = PrioritizedReplayBuffer(MEMORY_SIZE)
        
        # Training metrics
        self.best_score = 0
        self.training_stats = {
            'episode_rewards': [],
            'episode_lengths': [],
            'episode_losses': [],
            'best_scores': []
        }
        
        # Additional v5 components
        self.curriculum_stage = 0
        self.stage_scores = [10, 20, 30, 40, 50]
        self.mcts_simulations = 50
        self.reward_history = deque(maxlen=1000)
        self.reward_std = 1.0
        
        # Load model if exists
        self.load_model()
        
        # Initialize Hamiltonian cycle
        self.hamiltonian_cycle = self.generate_hamiltonian_cycle()

    def start_training(self, num_episodes=EPISODES):
        """Main training loop with enhanced features"""
        for episode in range(self.episode, num_episodes + 1):
            self.reset_game_state()
            game_over = False
            episode_start_time = time.time()
            episode_reward = 0
            
            while not game_over:
                state = self.get_state()
                graph_data = self.create_graph_state()
                old_distance = calculate_distance(self.snake_coords[0], self.food_coord)
                
                # Enhanced action selection with MCTS
                if random.random() < 0.3:  # 30% chance for rule-based action
                    action = self.get_rule_based_action()
                else:
                    action = self.select_action_with_mcts(state, graph_data)
                
                # Execute action
                self.update_direction(action)
                self.move_snake()
                
                # Check game status
                game_over = self.check_collision(self.snake_coords[0])
                ate_food = self.snake_coords[0] == self.food_coord
                new_distance = calculate_distance(self.snake_coords[0], self.food_coord)
                
                # Enhanced reward calculation
                reward = self.calculate_reward(old_distance, new_distance, ate_food, game_over)
                episode_reward += reward
                
                # Update game state
                if ate_food:
                    self.score += 1
                    self.steps_without_food = 0
                    self.food_coord = self.place_food()
                else:
                    del self.snake_coords[-1]
                    self.steps_without_food += 1
                
                # Store experience and train
                next_state = self.get_state()
                next_graph_data = self.create_graph_state()
                self.memory.push((state, action, reward, next_state, game_over, graph_data, next_graph_data))
                
                if len(self.memory.buffer) >= BATCH_SIZE:
                    self.train()
                
                # Update curriculum and learning parameters
                self.update_curriculum()
            
            # Episode finished
            self.training_stats['episode_rewards'].append(self.score)
            self.training_stats['episode_lengths'].append(self.total_steps)
            
            # Update best score
            if self.score > self.best_score:
                self.best_score = self.score
                self.training_stats['best_scores'].append((episode, self.best_score))
            
            # Periodic saving and reporting
            if episode % 100 == 0:
                self.save_model()
                avg_score = sum(self.training_stats['episode_rewards'][-100:]) / 100
                episode_time = time.time() - episode_start_time
                print(f"Episode: {episode}, Score: {self.score}, Best Score: {self.best_score}, "
                      f"Average Score (last 100): {avg_score:.2f}, "
                      f"Curriculum Stage: {self.curriculum_stage}, "
                      f"Time: {episode_time:.2f}s")
            
            self.episode += 1

    def save_model(self):
        """Save model with additional v5 components"""
        save_data = {
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'episode': self.episode,
            'epsilon': EPSILON,
            'best_score': self.best_score,
            'training_stats': self.training_stats,
            'curriculum_stage': self.curriculum_stage,
            'reward_std': self.reward_std
        }
        torch.save(save_data, SAVE_PATH)
        print(f"Model and training stats saved to {SAVE_PATH}")

    def load_model(self):
        """Load model with additional v5 components"""
        if os.path.exists(SAVE_PATH):
            try:
                checkpoint = torch.load(SAVE_PATH)
                self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
                self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.episode = checkpoint.get('episode', 1)
                self.best_score = checkpoint.get('best_score', 0)
                self.curriculum_stage = checkpoint.get('curriculum_stage', 0)
                self.reward_std = checkpoint.get('reward_std', 1.0)
                self.training_stats = checkpoint.get('training_stats', {
                    'episode_rewards': [],
                    'episode_lengths': [],
                    'episode_losses': [],
                    'best_scores': []
                })
                
                global EPSILON
                EPSILON = checkpoint.get('epsilon', EPSILON)
            except Exception as e:
                print(f"Error loading model: {e}")
                self.episode = 1
                self.best_score = 0

    def train(self):
        if len(self.memory.buffer) < BATCH_SIZE:
            return

        # Sample from prioritized replay buffer
        samples, indices, weights = self.memory.sample(BATCH_SIZE)
        if samples is None:
            return
            
        states, actions, rewards, next_states, dones, graph_states, next_graph_states = zip(*samples)
        weights = torch.FloatTensor(weights).to(self.device)
        
        # Convert to tensors
        states = torch.stack(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.stack(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Get current and next distributions
        current_dist = self.policy_net(states, return_distribution=True)
        next_dist = self.target_net(next_states, return_distribution=True)
        
        # Compute distributional loss
        proj_dist = self.distribute_bellman(next_dist, rewards, dones)
        dist_loss = -(proj_dist * current_dist.log()).sum(-1)
        
        # Weight losses by importance sampling weights
        weighted_loss = (dist_loss * weights).mean()
        
        # Optimize
        self.optimizer.zero_grad()
        weighted_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        # Update priorities
        priorities = dist_loss.detach().cpu().numpy()
        self.memory.update_priorities(indices, priorities)
        
        # Reset noise for next forward pass
        self.policy_net.reset_noise()
        self.target_net.reset_noise()

    def distribute_bellman(self, next_dist, rewards, dones):
        batch_size = next_dist.size(0)
        
        # Compute projected values
        projected_values = rewards.unsqueeze(-1) + GAMMA * \
                         (1 - dones.unsqueeze(-1)) * \
                         self.supports.unsqueeze(0)
                         
        # Clamp values to support range
        projected_values.clamp_(self.v_min, self.v_max)
        
        # Compute projection
        delta_z = (self.v_max - self.v_min) / (self.n_atoms - 1)
        b = (projected_values - self.v_min) / delta_z
        l = b.floor().long()
        u = b.ceil().long()
        
        # Distribute probability mass
        proj_dist = torch.zeros_like(next_dist)
        offset = torch.linspace(0, (batch_size - 1) * self.n_atoms, batch_size).long()\
                .unsqueeze(1).expand(batch_size, self.n_atoms)
        
        proj_dist.view(-1).index_add_(0, (offset + l).view(-1),
                                    (next_dist * (u.float() - b)).view(-1))
        proj_dist.view(-1).index_add_(0, (offset + u).view(-1),
                                    (next_dist * (b - l.float())).view(-1))
        
        return proj_dist

    def get_dynamic_epsilon(self):
        """Adaptive exploration rate based on performance"""
        recent_scores = self.training_stats['episode_rewards'][-100:] if self.training_stats['episode_rewards'] else [0]
        avg_score = sum(recent_scores) / len(recent_scores)
        
        # Reduce exploration as performance improves
        dynamic_epsilon = max(
            EPSILON_MIN,
            EPSILON * math.exp(-0.01 * avg_score)
        )
        return dynamic_epsilon

    def select_action_with_mcts(self, state, graph_data):
        """Action selection using MCTS with neural network guidance"""
        if random.random() < self.get_dynamic_epsilon():
            return random.randint(0, 3)

        root = MCTSNode(state)
        
        # Perform MCTS simulations
        for _ in range(self.mcts_simulations):
            node = root
            sim_state = state.clone()
            sim_graph = copy.deepcopy(graph_data)
            done = False
            total_reward = 0
            
            # Selection
            while not done and not node.untried_actions:
                action, child = node.select_child()
                sim_state, reward, done = self.simulate_step(sim_state, action)
                total_reward += reward
                node = child
            
            # Expansion
            if not done and node.untried_actions:
                action = random.choice(node.untried_actions)
                node.untried_actions.remove(action)
                sim_state, reward, done = self.simulate_step(sim_state, action)
                node = node.expand(action, sim_state)
                total_reward += reward
            
            # Simulation
            while not done:
                with torch.no_grad():
                    action = self.policy_net(sim_state, sim_graph).argmax().item()
                sim_state, reward, done = self.simulate_step(sim_state, action)
                total_reward += reward
            
            # Backpropagation
            while node:
                node.update(total_reward)
                node = node.parent
        
        # Select best action from root
        return max(root.children.items(),
                  key=lambda item: item[1].visits)[0]

    def simulate_step(self, state, action):
        """Simulate a step for MCTS"""
        # Create a copy of the game state
        temp_snake = copy.deepcopy(self.snake_coords)
        temp_direction = self.direction
        temp_food = copy.deepcopy(self.food_coord)
        
        # Update direction
        if action == 0 and temp_direction != 'down': temp_direction = 'up'
        elif action == 1 and temp_direction != 'up': temp_direction = 'down'
        elif action == 2 and temp_direction != 'right': temp_direction = 'left'
        elif action == 3 and temp_direction != 'left': temp_direction = 'right'
        
        # Move snake
        head = temp_snake[0]
        new_head = [head[0], head[1]]
        if temp_direction == 'up': new_head[1] -= SPACE_SIZE
        elif temp_direction == 'down': new_head[1] += SPACE_SIZE
        elif temp_direction == 'left': new_head[0] -= SPACE_SIZE
        elif temp_direction == 'right': new_head[0] += SPACE_SIZE
        
        # Check collision
        done = (new_head[0] < 0 or new_head[0] >= WIDTH or
                new_head[1] < 0 or new_head[1] >= HEIGHT or
                new_head in temp_snake[1:])
        
        if done:
            return state, -1, True
        
        # Check food
        reward = 0
        if new_head == temp_food:
            reward = 1
        else:
            temp_snake.pop()
        
        temp_snake.insert(0, new_head)
        
        # Create new state
        new_state = self.create_state_from_simulation(temp_snake, temp_food, temp_direction)
        return new_state, reward, done

    def create_state_from_simulation(self, snake, food, direction):
        """Create state tensor from simulated game state"""
        head = snake[0]
        point_l = [head[0] - SPACE_SIZE, head[1]]
        point_r = [head[0] + SPACE_SIZE, head[1]]
        point_u = [head[0], head[1] - SPACE_SIZE]
        point_d = [head[0], head[1] + SPACE_SIZE]
        
        state = [
            direction == 'up', direction == 'down',
            direction == 'left', direction == 'right',
            head[0] < food[0], head[0] > food[0],
            head[1] < food[1], head[1] > food[1],
            self.check_collision_sim(point_u, snake),
            self.check_collision_sim(point_d, snake),
            self.check_collision_sim(point_l, snake),
            self.check_collision_sim(point_r, snake)
        ]
        return torch.FloatTensor(state).to(self.device)

    def check_collision_sim(self, position, snake):
        """Check collision for simulation"""
        x, y = position
        if x < 0 or x >= WIDTH or y < 0 or y >= HEIGHT:
            return True
        return position in snake[1:]

    def update_curriculum(self):
        """Update curriculum stage based on performance"""
        if self.curriculum_stage >= len(self.stage_scores):
            return
            
        recent_scores = self.training_stats['episode_rewards'][-50:]
        if len(recent_scores) < 50:
            return
            
        avg_score = sum(recent_scores) / len(recent_scores)
        if avg_score >= self.stage_scores[self.curriculum_stage]:
            self.curriculum_stage += 1
            # Adjust parameters for next stage
            global LEARNING_RATE
            LEARNING_RATE *= 0.8
            self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)

    def calculate_reward(self, old_distance, new_distance, ate_food, game_over):
        """Enhanced reward calculation with curriculum learning"""
        reward = 0
        head = self.snake_coords[0]
        
        # Base rewards/penalties with length and curriculum scaling
        if game_over:
            reward = -150 * (1 + len(self.snake_coords) / 8)  # Increased penalty for dying
        elif ate_food:
            # Reward for eating food scales with snake length, efficiency, and curriculum stage
            base_food_reward = 100
            length_bonus = len(self.snake_coords) * 3
            efficiency_bonus = max(0, 50 - self.steps_without_food) * 2
            curriculum_bonus = self.curriculum_stage * 10
            reward = base_food_reward + length_bonus + efficiency_bonus + curriculum_bonus
        
        # Distance and path optimization rewards
        if not game_over and not ate_food:
            # Distance-based reward with efficiency scaling
            distance_weight = 2.0 / (1.0 + len(self.snake_coords) / 15)
            if new_distance < old_distance:
                reward += 3 * distance_weight
            else:
                reward -= 2 * distance_weight
            
            # Reward for following optimal path
            path = self.bfs_path_to_food()
            if path:
                optimal_length = len(path)
                # Avoid division by zero and reward efficiency
                path_efficiency = 1.0 / max(1.0, abs(self.steps_without_food - optimal_length + 1))
                reward += 2 * path_efficiency
                
                # Additional reward for being on the optimal path
                if list(path[1]) == self.snake_coords[0]:
                    reward += 1
        
        # Space utilization rewards
        available_space = self.flood_fill(head)
        min_required_space = len(self.snake_coords) + 5
        space_reward = (available_space - min_required_space) * 0.2
        reward += max(space_reward, 0)
        
        # Survival rewards based on length and space
        survival_bonus = 0.2 * math.sqrt(len(self.snake_coords))
        if available_space > min_required_space * 2:
            survival_bonus *= 1.5  # Bonus for maintaining good space
        reward += survival_bonus
        
        # Penalties for risky behavior
        dangers = self.check_immediate_dangers()
        danger_count = sum(1 for d in dangers.values() if d)
        if danger_count > 2:  # Penalize being in tight spots
            reward -= 5 * danger_count
        
        # Curriculum-based reward scaling
        curriculum_multiplier = 1.0 + (self.curriculum_stage * 0.2)
        reward *= curriculum_multiplier
        
        # Add to reward history for normalization
        self.reward_history.append(reward)
        if len(self.reward_history) > 100:
            self.reward_std = np.std(self.reward_history) or 1.0
        
        # Normalize reward
        normalized_reward = reward / (self.reward_std + 1e-8)  # Add small epsilon to avoid division by zero
        
        return normalized_reward

    def train_episode(self):
        """Enhanced training episode with all improvements"""
        self.reset_game_state()
        game_over = False
        episode_reward = 0
        
        while not game_over:
            state = self.get_state()
            graph_data = self.create_graph_state()
            
            # Use MCTS for action selection
            action = self.select_action_with_mcts(state, graph_data)
            
            # Execute action and get reward
            old_distance = calculate_distance(self.snake_coords[0], self.food_coord)
            self.update_direction(action)
            self.move_snake()
            
            # Check game status
            game_over = self.check_collision(self.snake_coords[0])
            ate_food = self.snake_coords[0] == self.food_coord
            new_distance = calculate_distance(self.snake_coords[0], self.food_coord)
            
            # Calculate reward with curriculum learning
            reward = self.calculate_reward(old_distance, new_distance, ate_food, game_over)
            episode_reward += reward
            
            # Update game state
            if ate_food:
                self.score += 1
                self.steps_without_food = 0
                self.food_coord = self.place_food()
            else:
                del self.snake_coords[-1]
                self.steps_without_food += 1
            
            # Store experience and train
            next_state = self.get_state()
            next_graph_data = self.create_graph_state()
            self.memory.push((state, action, reward, next_state, game_over, graph_data, next_graph_data))
            
            if len(self.memory.buffer) >= BATCH_SIZE:
                self.train()
            
            # Update curriculum
            self.update_curriculum()
        
        return episode_reward

    def reset_game_state(self):
        """Reset game state"""
        self.score = 0
        self.direction = 'right'
        self.snake_coords = [[100, 100], [80, 100], [60, 100]]
        self.food_coord = self.place_food()
        self.steps_without_food = 0
        self.total_steps += 1

    def move_snake(self):
        """Update snake position based on direction"""
        x, y = self.snake_coords[0]
        if self.direction == 'up':
            y -= SPACE_SIZE
        elif self.direction == 'down':
            y += SPACE_SIZE
        elif self.direction == 'left':
            x -= SPACE_SIZE
        elif self.direction == 'right':
            x += SPACE_SIZE
        self.snake_coords.insert(0, [x, y])

    def update_direction(self, action):
        """Update snake direction based on action"""
        if action == 0 and self.direction != 'down':  # Up
            self.direction = 'up'
        elif action == 1 and self.direction != 'up':  # Down
            self.direction = 'down'
        elif action == 2 and self.direction != 'right':  # Left
            self.direction = 'left'
        elif action == 3 and self.direction != 'left':  # Right
            self.direction = 'right'

    def place_food(self):
        """Place food in random empty location"""
        while True:
            x = random.randint(0, (WIDTH // SPACE_SIZE) - 1) * SPACE_SIZE
            y = random.randint(0, (HEIGHT // SPACE_SIZE) - 1) * SPACE_SIZE
            if [x, y] not in self.snake_coords:
                return [x, y]

    def check_collision(self, position):
        """Check if position collides with walls or snake body"""
        x, y = position
        if x < 0 or x >= WIDTH or y < 0 or y >= HEIGHT:
            return True
        return position in self.snake_coords[1:]

    def get_state(self):
        """Get current state representation"""
        head = self.snake_coords[0]
        point_l = [head[0] - SPACE_SIZE, head[1]]
        point_r = [head[0] + SPACE_SIZE, head[1]]
        point_u = [head[0], head[1] - SPACE_SIZE]
        point_d = [head[0], head[1] + SPACE_SIZE]
        
        dir_l = self.direction == 'left'
        dir_r = self.direction == 'right'
        dir_u = self.direction == 'up'
        dir_d = self.direction == 'down'
        
        state = [
            dir_u,     # Direction up
            dir_d,     # Direction down
            dir_l,     # Direction left
            dir_r,     # Direction right
            head[0] < self.food_coord[0],  # Food right
            head[0] > self.food_coord[0],  # Food left
            head[1] < self.food_coord[1],  # Food down
            head[1] > self.food_coord[1],  # Food up
            self.check_collision(point_u),  # Collision up
            self.check_collision(point_d),  # Collision down
            self.check_collision(point_l),  # Collision left
            self.check_collision(point_r)   # Collision right
        ]
        
        return torch.FloatTensor([1 if x else 0 for x in state]).to(self.device)

    def create_graph_state(self):
        """Create graph representation of current state"""
        try:
            nodes = self.snake_coords + [self.food_coord]
            node_features = []
            
            for node in nodes:
                features = [
                    float(node == self.snake_coords[0]),  # Is head
                    float(node == self.food_coord),  # Is food
                    float(node in self.snake_coords[1:]),  # Is body
                    1.0  # Constant feature
                ]
                node_features.append(features)
            
            edges = []
            for i in range(len(self.snake_coords) - 1):
                edges.extend([[i, i+1], [i+1, i]])
            
            edges.extend([[0, len(nodes)-1], [len(nodes)-1, 0]])
            
            if not edges:
                edges = [[0, 0]]
            
            x = torch.FloatTensor(node_features).to(self.device)
            edge_index = torch.LongTensor(edges).t().to(self.device)
            
            return Data(x=x, edge_index=edge_index)
        except Exception as e:
            print(f"Graph state creation error: {e}")
            return Data(
                x=torch.ones(1, 4).to(self.device),
                edge_index=torch.LongTensor([[0], [0]]).to(self.device)
            )

    def generate_hamiltonian_cycle(self):
        """Generate a Hamiltonian cycle for the grid"""
        width_cells = WIDTH // SPACE_SIZE
        height_cells = HEIGHT // SPACE_SIZE
        cycle = []
        
        for y in range(height_cells):
            row = range(width_cells) if y % 2 == 0 else range(width_cells - 1, -1, -1)
            for x in row:
                cycle.append([x * SPACE_SIZE, y * SPACE_SIZE])
        
        return cycle

    def bfs_path_to_food(self):
        """Find shortest path to food using BFS"""
        start = tuple(self.snake_coords[0])
        goal = tuple(self.food_coord)
        queue = Queue()
        queue.put(start)
        visited = {start: None}
        
        while not queue.empty():
            current = queue.get()
            if current == goal:
                break
                
            # Check all possible moves
            for dx, dy in [(0, -SPACE_SIZE), (0, SPACE_SIZE), (-SPACE_SIZE, 0), (SPACE_SIZE, 0)]:
                next_pos = (current[0] + dx, current[1] + dy)
                if (next_pos not in visited and
                    0 <= next_pos[0] < WIDTH and
                    0 <= next_pos[1] < HEIGHT and
                    list(next_pos) not in self.snake_coords[1:]):
                    queue.put(next_pos)
                    visited[next_pos] = current
                    
        if goal not in visited:
            return None
            
        # Reconstruct path
        path = []
        current = goal
        while current is not None:
            path.append(current)
            current = visited[current]
        return path[::-1]

    def flood_fill(self, start_pos):
        """Count reachable cells using flood fill algorithm"""
        visited = set()
        queue = [tuple(start_pos)]
        
        while queue:
            current = queue.pop(0)
            if current not in visited:
                visited.add(current)
                x, y = current
                
                for dx, dy in [(0, SPACE_SIZE), (0, -SPACE_SIZE), (SPACE_SIZE, 0), (-SPACE_SIZE, 0)]:
                    next_pos = (x + dx, y + dy)
                    if (0 <= next_pos[0] < WIDTH and 
                        0 <= next_pos[1] < HEIGHT and 
                        list(next_pos) not in self.snake_coords[1:] and 
                        next_pos not in visited):
                        queue.append(next_pos)
                        
        return len(visited)

    def check_immediate_dangers(self):
        """Check dangers in all directions"""
        head = self.snake_coords[0]
        dangers = {
            'up': False, 'down': False, 'left': False, 'right': False
        }
        
        # Check each direction
        positions = {
            'up': [head[0], head[1] - SPACE_SIZE],
            'down': [head[0], head[1] + SPACE_SIZE],
            'left': [head[0] - SPACE_SIZE, head[1]],
            'right': [head[0] + SPACE_SIZE, head[1]]
        }
        
        for direction, pos in positions.items():
            # Check wall collision
            if (pos[0] < 0 or pos[0] >= WIDTH or
                pos[1] < 0 or pos[1] >= HEIGHT):
                dangers[direction] = True
                continue
                
            # Check self collision
            if pos in self.snake_coords[1:]:
                dangers[direction] = True
                
        return dangers

    def get_rule_based_action(self):
        """Get action based on rules and current game state"""
        head = self.snake_coords[0]
        dangers = self.check_immediate_dangers()
        
        # Get available space using flood fill
        available_space = self.flood_fill(head)
        min_safe_space = len(self.snake_coords) + 5
        
        # Try to find path to food using BFS
        path = self.bfs_path_to_food()
        
        # If snake is long enough, occasionally follow Hamiltonian cycle
        if len(self.snake_coords) > (WIDTH // SPACE_SIZE) * (HEIGHT // SPACE_SIZE) // 3:
            if random.random() < 0.3:
                next_pos = self.get_hamiltonian_next_move()
                dx = next_pos[0] - head[0]
                dy = next_pos[1] - head[1]
                
                if dx > 0 and not dangers['right']: return 3
                if dx < 0 and not dangers['left']: return 2
                if dy > 0 and not dangers['down']: return 1
                if dy < 0 and not dangers['up']: return 0
        
        # If path to food exists and enough space available
        if path and available_space > min_safe_space:
            next_pos = path[1]
            dx = next_pos[0] - head[0]
            dy = next_pos[1] - head[1]
            
            # Verify move safety using flood fill
            if dx > 0 and not dangers['right']:
                test_space = self.flood_fill([head[0] + SPACE_SIZE, head[1]])
                if test_space > min_safe_space: return 3
            if dx < 0 and not dangers['left']:
                test_space = self.flood_fill([head[0] - SPACE_SIZE, head[1]])
                if test_space > min_safe_space: return 2
            if dy < 0 and not dangers['up']:
                test_space = self.flood_fill([head[0], head[1] - SPACE_SIZE])
                if test_space > min_safe_space: return 0
            if dy > 0 and not dangers['down']:
                test_space = self.flood_fill([head[0], head[1] + SPACE_SIZE])
                if test_space > min_safe_space: return 1
        
        # If no safe path to food, maximize available space
        max_space = 0
        best_action = self.follow_tail()
        
        for direction, danger in dangers.items():
            if not danger:
                test_pos = list(head)
                if direction == 'right': test_pos[0] += SPACE_SIZE
                elif direction == 'left': test_pos[0] -= SPACE_SIZE
                elif direction == 'up': test_pos[1] -= SPACE_SIZE
                elif direction == 'down': test_pos[1] += SPACE_SIZE
                
                space = self.flood_fill(test_pos)
                if space > max_space:
                    max_space = space
                    best_action = self.direction_to_action(direction)
        
        return best_action

    def follow_tail(self):
        """Follow tail when no better option is available"""
        tail = self.snake_coords[-1]
        head = self.snake_coords[0]
        
        if tail[0] < head[0] and self.direction != 'right': return 2  # LEFT
        if tail[0] > head[0] and self.direction != 'left': return 3   # RIGHT
        if tail[1] < head[1] and self.direction != 'down': return 0   # UP
        if tail[1] > head[1] and self.direction != 'up': return 1     # DOWN
        
        return random.randint(0, 3)

    def direction_to_action(self, direction):
        """Convert direction string to action number"""
        return {'up': 0, 'down': 1, 'left': 2, 'right': 3}[direction]

    def get_hamiltonian_next_move(self):
        """Get next move based on Hamiltonian cycle"""
        head_pos = self.snake_coords[0]
        try:
            current_idx = self.hamiltonian_cycle.index(head_pos)
            next_idx = (current_idx + 1) % len(self.hamiltonian_cycle)
            return self.hamiltonian_cycle[next_idx]
        except ValueError:
            # If head is not in cycle, return closest cycle point
            min_dist = float('inf')
            closest_point = self.hamiltonian_cycle[0]
            for point in self.hamiltonian_cycle:
                dist = calculate_distance(head_pos, point)
                if dist < min_dist:
                    min_dist = dist
                    closest_point = point
            return closest_point

def calculate_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

game = EnhancedSnakeGameAI()
game.start_training()
