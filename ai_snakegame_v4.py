import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
from collections import deque, defaultdict
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from queue import Queue
import torch.nn.init as init

# Game configuration
WIDTH = 300
HEIGHT = 300
SPACE_SIZE = 20
BODY_SIZE = 2

# RL parameters
EPISODES = 5000  # Increased from 2000
GAMMA = 0.999  # Increased for better long-term planning
LEARNING_RATE = 5e-5  # Further reduced for stability
EPSILON = 1.0
EPSILON_DECAY = 0.9995  # Adjusted for longer exploration
EPSILON_MIN = 0.01  # Decreased minimum epsilon
BATCH_SIZE = 1024  # Increased for better gradient estimates
MEMORY_SIZE = 100000  # Increased from 50000
SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
SAVE_PATH = os.path.join(SAVE_DIR, "modelv4.pt")
UPDATE_TARGET_EVERY = 2  # More frequent updates

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return self.norm(x)

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.attention = MultiHeadAttention(dim, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
            nn.LayerNorm(dim)
        )
        
    def forward(self, x):
        x = x + self.attention(x)
        x = x + self.ffn(x)
        return x

class EnhancedGNNStateEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.attention = MultiHeadAttention(hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.gelu(self.conv1(x, edge_index))
        x = F.gelu(self.conv2(x, edge_index))
        x = x.unsqueeze(0)  # Add batch dimension
        x = self.attention(x)
        x = self.norm(x.squeeze(0))
        return torch.mean(x, dim=0)

class DQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.gnn = EnhancedGNNStateEncoder(input_dim=4, hidden_dim=128)
        
        # Enhanced architecture
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        
        # Transformer blocks for feature processing
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(512, num_heads=8)
            for _ in range(3)
        ])
        
        # Separate streams with enhanced processing
        self.value_stream = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 1)
        )
        
        self.advantage_stream = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, action_dim)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            init.orthogonal_(module.weight.data)
            if module.bias is not None:
                init.constant_(module.bias.data, 0)
                
    def forward(self, state, graph_data=None):
        batch_size = state.size(0) if len(state.size()) > 1 else 1
        if len(state.size()) == 1:
            state = state.unsqueeze(0)
            
        if graph_data is not None:
            gnn_features = self.gnn(graph_data)
            if len(gnn_features.shape) == 1:
                gnn_features = gnn_features.unsqueeze(0)
            gnn_features = gnn_features.expand(batch_size, -1)
            
            if state.size(1) != 12:
                raise ValueError(f"Unexpected state dimension: {state.size()}")
            x = torch.cat([state, gnn_features], dim=1)
        else:
            x = state
            
        # Initial feature processing
        x = self.state_encoder(x)
        
        # Add sequence dimension for transformer
        x = x.unsqueeze(1)
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
            
        # Remove sequence dimension
        x = x.squeeze(1)
        
        # Dueling streams
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        
        # Combine streams with normalized advantage
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values

class SnakeGameAI:
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
        
        # Initialize networks with correct dimensions
        state_dim = 12 + 128  # Update state dimension to match GNN output
        action_dim = 4
        self.policy_net = DQNetwork(state_dim, action_dim).to(self.device)
        self.target_net = DQNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.memory = deque(maxlen=MEMORY_SIZE)
        
        # Training metrics
        self.best_score = 0  # Add best score tracking
        self.training_stats = {
            'episode_rewards': [],
            'episode_lengths': [],
            'episode_losses': [],
            'best_scores': []  # Add best scores tracking
        }
        
        # Load model if exists
        self.load_model()
        
        # Initialize Hamiltonian cycle
        self.hamiltonian_cycle = self.generate_hamiltonian_cycle()
        self.current_cycle_pos = 0

    def reset_game_state(self):
        """Reset game state"""
        self.score = 0
        self.direction = 'right'
        self.snake_coords = [[100, 100], [80, 100], [60, 100]]
        self.food_coord = self.place_food()
        self.steps_without_food = 0
        self.total_steps += 1

    def start_training(self, num_episodes=EPISODES):
        """Main training loop"""
        for episode in range(self.episode, num_episodes + 1):
            self.reset_game_state()
            game_over = False
            episode_start_time = time.time()
            
            while not game_over:
                # Get current state
                state = self.get_state()
                graph_data = self.create_graph_state()
                old_distance = calculate_distance(self.snake_coords[0], self.food_coord)
                
                # Get action
                if random.random() < 0.3:  # 30% chance to use rule-based action
                    action = self.get_rule_based_action()
                else:
                    action = self.act(state, graph_data)
                
                # Update direction and move
                self.update_direction(action)
                self.move_snake()
                
                # Check game status
                game_over = self.check_collision(self.snake_coords[0])
                ate_food = self.snake_coords[0] == self.food_coord
                new_distance = calculate_distance(self.snake_coords[0], self.food_coord)
                
                # Calculate reward
                reward = self.calculate_reward(old_distance, new_distance, ate_food, game_over)
                
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
                self.memory.append((state, action, reward, next_state, game_over, graph_data, next_graph_data))
                
                if len(self.memory) >= BATCH_SIZE:
                    self.train()
                
                # Update exploration rate
                global EPSILON
                if EPSILON > EPSILON_MIN:
                    EPSILON *= EPSILON_DECAY
            
            # Episode finished
            self.training_stats['episode_rewards'].append(self.score)
            self.training_stats['episode_lengths'].append(self.total_steps)
            
            # Update best score
            if self.score > self.best_score:
                self.best_score = self.score
                self.training_stats['best_scores'].append((episode, self.best_score))
            
            if episode % 100 == 0:
                self.save_model()
                avg_score = sum(self.training_stats['episode_rewards'][-100:]) / 100
                episode_time = time.time() - episode_start_time
                print(f"Episode: {episode}, Score: {self.score}, Best Score: {self.best_score}, "
                      f"Average Score (last 100): {avg_score:.2f}, Epsilon: {EPSILON:.4f}, "
                      f"Time: {episode_time:.2f}s")
            
            self.episode += 1

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

    def create_graph_state(self):
        try:
            # Create node features
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
                
            # Create edges (connecting adjacent snake parts and food)
            edges = []
            for i in range(len(self.snake_coords) - 1):
                edges.extend([[i, i+1], [i+1, i]])  # Bidirectional edges
                
            # Connect head to food
            edges.extend([[0, len(nodes)-1], [len(nodes)-1, 0]])
            
            # Handle empty edge case
            if not edges:
                edges = [[0, 0]]  # Self-loop if no other edges
                
            x = torch.FloatTensor(node_features).to(self.device)
            edge_index = torch.LongTensor(edges).t().to(self.device)
            
            return Data(x=x, edge_index=edge_index)
        except Exception as e:
            print(f"Graph state creation error: {e}")
            # Return a minimal valid graph state
            return Data(
                x=torch.ones(1, 4).to(self.device),
                edge_index=torch.LongTensor([[0], [0]]).to(self.device)
            )

    def bfs_path_to_food(self):
        start = tuple(self.snake_coords[0])
        goal = tuple(self.food_coord)
        queue = Queue()
        queue.put(start)
        visited = {start: None}
        
        while not queue.empty():
            current = queue.get()
            if current == goal:
                break
                
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

    def generate_hamiltonian_cycle(self):
        """Generate a Hamiltonian cycle for the grid"""
        width_cells = WIDTH // SPACE_SIZE
        height_cells = HEIGHT // SPACE_SIZE
        cycle = []
        
        # Generate simple snake-like Hamiltonian cycle
        for y in range(height_cells):
            row = range(width_cells) if y % 2 == 0 else range(width_cells - 1, -1, -1)
            for x in row:
                cycle.append([x * SPACE_SIZE, y * SPACE_SIZE])
        
        return cycle

    def get_hamiltonian_next_move(self):
        """Get next move based on Hamiltonian cycle"""
        head_pos = self.snake_coords[0]
        current_idx = self.hamiltonian_cycle.index(head_pos)
        next_idx = (current_idx + 1) % len(self.hamiltonian_cycle)
        return self.hamiltonian_cycle[next_idx]

    def get_rule_based_action(self):
        head = self.snake_coords[0]
        food = self.food_coord
        dangers = self.check_immediate_dangers()
        
        # Get available space using flood fill
        available_space = self.flood_fill(head)
        min_safe_space = len(self.snake_coords) + 5  # Minimum safe space needed
        
        # Try to find path to food using BFS
        path = self.bfs_path_to_food()
        
        # If snake is long enough, occasionally follow Hamiltonian cycle
        if len(self.snake_coords) > (WIDTH // SPACE_SIZE) * (HEIGHT // SPACE_SIZE) // 3:
            if random.random() < 0.3:  # 30% chance to use Hamiltonian cycle
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
        best_action = self.follow_tail()  # Default to following tail
        
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

    def check_immediate_dangers(self):
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

    def direction_to_action(self, direction):
        return {'up': 0, 'down': 1, 'left': 2, 'right': 3}[direction]

    def follow_tail(self):
        # Simple tail-following logic
        tail = self.snake_coords[-1]
        head = self.snake_coords[0]
        
        if tail[0] < head[0] and self.direction != 'right': return 2  # LEFT
        if tail[0] > head[0] and self.direction != 'left': return 3   # RIGHT
        if tail[1] < head[1] and self.direction != 'down': return 0   # UP
        if tail[1] > head[1] and self.direction != 'up': return 1     # DOWN
        
        return random.randint(0, 3)

    def calculate_reward(self, old_distance, new_distance, ate_food, game_over):
        reward = 0
        head = self.snake_coords[0]
        
        # Base rewards/penalties with length scaling
        if game_over:
            reward = -150 * (1 + len(self.snake_coords) / 8)  # Increased penalty for dying
        elif ate_food:
            # Reward for eating food scales with snake length and efficiency
            base_food_reward = 100
            length_bonus = len(self.snake_coords) * 3
            efficiency_bonus = max(0, 50 - self.steps_without_food) * 2
            reward = base_food_reward + length_bonus + efficiency_bonus
        
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
        
        # Hamiltonian cycle alignment reward for long snakes
        if len(self.snake_coords) > (WIDTH // SPACE_SIZE) * (HEIGHT // SPACE_SIZE) // 3:
            next_hamiltonian = self.get_hamiltonian_next_move()
            if self.snake_coords[0] == next_hamiltonian:
                reward += 2  # Reward for following Hamiltonian cycle
        
        # Time penalty scales with snake length but is capped
        steps_penalty = min(
            5.0,  # Cap the maximum penalty
            (self.steps_without_food * 0.02) * (1 + len(self.snake_coords) / 15)
        )
        reward -= steps_penalty
        
        return reward

    def get_state(self):
        snake_head = self.snake_coords[0]
        point_l = [snake_head[0] - SPACE_SIZE, snake_head[1]]
        point_r = [snake_head[0] + SPACE_SIZE, snake_head[1]]
        point_u = [snake_head[0], snake_head[1] - SPACE_SIZE]
        point_d = [snake_head[0], snake_head[1] + SPACE_SIZE]
        
        state = [
            self.direction == 'up', self.direction == 'down',
            self.direction == 'left', self.direction == 'right',
            snake_head[0] < self.food_coord[0],  # Food to the right
            snake_head[0] > self.food_coord[0],  # Food to the left
            snake_head[1] < self.food_coord[1],  # Food below
            snake_head[1] > self.food_coord[1],  # Food above
            self.check_collision(point_u),  # Collision up
            self.check_collision(point_d),  # Collision down
            self.check_collision(point_l),  # Collision left
            self.check_collision(point_r)   # Collision right
        ]
        return torch.FloatTensor(state).to(self.device)

    def act(self, state, graph_data):
        try:
            if random.random() <= EPSILON:
                return random.randint(0, 3)
            
            with torch.no_grad():
                state = state.unsqueeze(0)
                q_values = self.policy_net(state, graph_data)
                return q_values.argmax().item()
        except Exception as e:
            print(f"Action selection error: {e}")
            return random.randint(0, 3)

    def train(self):
        if len(self.memory) < BATCH_SIZE:
            return

        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones, graph_states, next_graph_states = zip(*batch)

        # Prepare tensors
        states = torch.stack(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.stack(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        try:
            # Convert graph states to tensors
            reference_graph_state = graph_states[0]
            reference_next_graph_state = next_graph_states[0]

            # Double DQN implementation
            with torch.no_grad():
                # Get actions from policy network
                next_actions = self.policy_net(next_states, reference_next_graph_state).argmax(1).unsqueeze(1)
                # Get Q-values from target network
                next_q_values = self.target_net(next_states, reference_next_graph_state).gather(1, next_actions)
                # Compute target Q-values
                target_q_values = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * GAMMA * next_q_values

            # Get current Q-values
            current_q_values = self.policy_net(states, reference_graph_state).gather(1, actions.unsqueeze(1))

            # Huber loss for more stable training
            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
            self.optimizer.step()

            # Store training statistics
            self.training_stats['episode_losses'].append(loss.item())

            # Update target network
            if self.episode % UPDATE_TARGET_EVERY == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

        except Exception as e:
            print(f"Training error: {e}")
            print(f"States shape: {states.shape}")
            print(f"Actions shape: {actions.shape}")
            return

    def update_direction(self, action):
        if action == 0 and self.direction != 'down':  # Up
            self.direction = 'up'
        elif action == 1 and self.direction != 'up':  # Down
            self.direction = 'down'
        elif action == 2 and self.direction != 'right':  # Left
            self.direction = 'left'
        elif action == 3 and self.direction != 'left':  # Right
            self.direction = 'right'

    def place_food(self):
        while True:
            x = random.randint(0, (WIDTH // SPACE_SIZE) - 1) * SPACE_SIZE
            y = random.randint(0, (HEIGHT // SPACE_SIZE) - 1) * SPACE_SIZE
            if [x, y] not in self.snake_coords:
                break
        return [x, y]

    def check_collision(self, position):
        x, y = position
        if x < 0 or x >= WIDTH or y < 0 or y >= HEIGHT:
            return True
        return position in self.snake_coords[1:]

    def save_model(self):
        """Save model state and training statistics"""
        os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
        
        save_data = {
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'episode': self.episode,
            'epsilon': EPSILON,
            'best_score': self.best_score,  # Add best score to saved data
            'training_stats': self.training_stats
        }
        
        torch.save(save_data, SAVE_PATH)
        print(f"Model and training stats saved to {SAVE_PATH}")

    def load_model(self):
        """Load model state and training statistics"""
        if os.path.exists(SAVE_PATH):
            try:
                checkpoint = torch.load(SAVE_PATH)
                self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
                self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.episode = checkpoint.get('episode', 1)
                self.best_score = checkpoint.get('best_score', 0)
                
                # Initialize training stats with default empty lists
                default_stats = {
                    'episode_rewards': [],
                    'episode_lengths': [],
                    'episode_losses': [],
                    'best_scores': []
                }
                self.training_stats = checkpoint.get('training_stats', default_stats)
                
                # Ensure best_scores exists in training_stats
                if 'best_scores' not in self.training_stats:
                    self.training_stats['best_scores'] = []
                
                global EPSILON
                EPSILON = checkpoint.get('epsilon', EPSILON)
            except Exception as e:
                print(f"Error loading model: {e}")
                self.episode = 1
                self.best_score = 0

def calculate_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


game = SnakeGameAI()
game.start_training()  # Start training immediately