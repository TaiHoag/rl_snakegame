import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import random
import numpy as np
import tensorflow as tf
import math
import json
from collections import deque
from tkinter import *

# Game configuration
WIDTH = 300
HEIGHT = 300
SPEED = 50  # Lower value = faster game
SPACE_SIZE = 20
BODY_SIZE = 2
SNAKE_COLOR = "#00FF00"
FOOD_COLOR = "#FFFFFF"
BACKGROUND_COLOR = "#000000"

# RL parameters
EPISODES = 1000
GAMMA = 0.95  # Discount rate
LEARNING_RATE = 0.001
EPSILON = 1.0  # Exploration rate
EPSILON_DECAY = 0.997
EPSILON_MIN = 0.05
BATCH_SIZE = 128
MEMORY_SIZE = 2000
SAVE_PATH = "snakegame/checkpoint.weights.h5"
SAVE_DIR = os.path.dirname(SAVE_PATH)

class SnakeGameAI:
    def __init__(self):
        self.window = Tk()
        self.window.title("Snake Game AI")

        self.score = 0
        self.direction = 'right'
        self.snake_coords = [[100, 100], [80, 100], [60, 100]]
        self.food_coord = self.place_food()

        self.canvas = Canvas(self.window, bg=BACKGROUND_COLOR, height=HEIGHT, width=WIDTH)
        self.canvas.pack()
        self.label = Label(self.window, text=f"Score: {self.score}", font=('consolas', 20))
        self.label.pack()

        self.snake_squares = [self.create_square(x, y) for x, y in self.snake_coords]
        self.food_square = self.create_oval(*self.food_coord)

        # Grid size based on the canvas and the space size
        self.grid_width = WIDTH // SPACE_SIZE
        self.grid_height = HEIGHT // SPACE_SIZE

        self.model = self.create_model()
        self.memory = deque(maxlen=MEMORY_SIZE)

        if not os.path.exists(SAVE_DIR):
            os.makedirs(SAVE_DIR)

        latest_checkpoint = tf.train.latest_checkpoint(SAVE_DIR)
        if latest_checkpoint:
            try:
                self.model.load_weights(latest_checkpoint).expect_partial()
                self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss='mse')  # Reset optimizer
                print(f"Successfully loaded the model weights from {latest_checkpoint}")
            except Exception as e:
                print(f"Error loading model: {e}")
                
        # Start the first episode
        self.episode = 1
        self.step_counter = 0
        self.run_game()

    def place_food(self):
        while True:
            x = random.randint(0, (WIDTH // SPACE_SIZE) - 1) * SPACE_SIZE
            y = random.randint(0, (HEIGHT // SPACE_SIZE) - 1) * SPACE_SIZE
            if [x, y] not in self.snake_coords:
                break
        return [x, y]

    def create_square(self, x, y):
        return self.canvas.create_rectangle(x, y, x + SPACE_SIZE, y + SPACE_SIZE, fill=SNAKE_COLOR)

    def create_oval(self, x, y):
        return self.canvas.create_oval(x, y, x + SPACE_SIZE, y + SPACE_SIZE, fill=FOOD_COLOR)

    def create_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, input_shape=(12,), activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(4, activation='linear')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss='mse')
        return model

    def get_state(self):
        snake_head = self.snake_coords[0]
        state = [
            self.direction == 'up', self.direction == 'down',
            self.direction == 'left', self.direction == 'right',
            snake_head[0] < self.food_coord[0],  # Food to the right
            snake_head[0] > self.food_coord[0],  # Food to the left
            snake_head[1] < self.food_coord[1],  # Food below
            snake_head[1] > self.food_coord[1],  # Food above
            self.check_collision([snake_head[0], snake_head[1] - SPACE_SIZE]),  # Collision up
            self.check_collision([snake_head[0], snake_head[1] + SPACE_SIZE]),  # Collision down
            self.check_collision([snake_head[0] - SPACE_SIZE, snake_head[1]]),  # Collision left
            self.check_collision([snake_head[0] + SPACE_SIZE, snake_head[1]])   # Collision right
        ]
        return np.array(state, dtype=int)

    def act(self, state):
        if np.random.rand() <= EPSILON:
            return random.choice([0, 1, 2, 3])
        q_values = self.model.predict(state.reshape(1, -1), verbose=0)
        return np.argmax(q_values[0])

    def replay(self):
        if len(self.memory) < BATCH_SIZE:
            return  # Only train when memory has enough experiences

        minibatch = random.sample(self.memory, BATCH_SIZE)
        states = np.array([s[0] for s in minibatch])
        actions = np.array([s[1] for s in minibatch])
        rewards = np.array([s[2] for s in minibatch])
        next_states = np.array([s[3] for s in minibatch])
        dones = np.array([s[4] for s in minibatch])

        q_values = self.model.predict(states, verbose=0)
        q_next = self.model.predict(next_states, verbose=0)

        for i in range(BATCH_SIZE):
            target = rewards[i]
            if not dones[i]:
                target += GAMMA * np.amax(q_next[i])
            q_values[i][actions[i]] = target

        # Train the model using the updated Q-values
        self.model.fit(states, q_values, epochs=1, verbose=0)

    def check_collision(self, position):
        x, y = position
        # Check wall collision
        if x < 0 or x >= WIDTH or y < 0 or y >= HEIGHT:
            return True
        # Check self collision
        for segment in self.snake_coords[1:]:
            if x == segment[0] and y == segment[1]:
                return True
        return False

    def run_game(self):
        global EPSILON
        
        state = self.get_state()
        snake_head = self.snake_coords[0]
        food_position = self.food_coord

        initial_distance = calculate_distance(snake_head, food_position)

        action = self.act(state)
        
        # Update direction based on action
        if action == 0 and self.direction != 'down':  # Up
            self.direction = 'up'
        elif action == 1 and self.direction != 'up':  # Down
            self.direction = 'down'
        elif action == 2 and self.direction != 'right':  # Left
            self.direction = 'left'
        elif action == 3 and self.direction != 'left':  # Right
            self.direction = 'right'
        
        self.move_snake()

        # Check for collisions (game over)
        if self.check_collision(self.snake_coords[0]):
            reward = -100
            next_state = self.get_state()
            self.memory.append((state, action, reward, next_state, True))
            
            self.replay()
            print(f"Episode: {self.episode}, Final Score: {self.score}")
            self.save_checkpoint()
            self.reset_game()
            self.episode += 1
            return

        new_distance = calculate_distance(self.snake_coords[0], self.food_coord)

        survival_reward = 0.2
        length_bonus = len(self.snake_coords) * 0.2

        reward = survival_reward  # Initial survival reward

        # Reward for moving closer to food
        if new_distance < initial_distance:
            reward += (initial_distance - new_distance) * 5
        else:
            reward -= (new_distance - initial_distance) * 3

        if (action == 0 and self.direction == 'down') or \
        (action == 1 and self.direction == 'up') or \
        (action == 2 and self.direction == 'right') or \
        (action == 3 and self.direction == 'left'):
            reward -= 2


        reward += self.evaluate_safety(snake_head)

        if self.snake_coords[0] == self.food_coord:
            reward = 25 + length_bonus
            self.score += 1
            self.label.config(text=f"Score: {self.score}")
            self.canvas.delete(self.food_square)
            self.food_coord = self.place_food()
            self.food_square = self.create_oval(*self.food_coord)
            print(f"Episode: {self.episode}, Score after eating food: {self.score}")
        else:
            self.remove_tail()  # Move the snake by removing the tail

        # Add state to memory for replay
        next_state = self.get_state()
        self.memory.append((state, action, reward, next_state, False))

        # Perform experience replay when memory is full
        if len(self.memory) >= BATCH_SIZE:
            self.replay()

        # Decay epsilon for exploration-exploitation balance
        if EPSILON > EPSILON_MIN:
            EPSILON *= EPSILON_DECAY

        # Schedule the next game loop iteration
        self.window.after(SPEED, self.run_game)

    def evaluate_safety(self, snake_head):
        """
        Evaluate how safe the current move is, based on proximity to walls,
        body, and available open space to avoid dead-ends and forced collisions.
        """
        penalty = 0
        x, y = snake_head

        # Penalize if snake is too close to the walls (within 1 step)
        if x < SPACE_SIZE or x >= WIDTH - SPACE_SIZE or y < SPACE_SIZE or y >= HEIGHT - SPACE_SIZE:
            penalty -= 1  # Penalize for being too close to the edges

        # Penalize if snake is too close to its own body
        if self.is_near_body(snake_head):
            penalty -= 1  # Penalize for moving closer to its own body

        # Reward for being in open space
        open_space_bonus = self.calculate_open_space(snake_head)
        penalty += open_space_bonus  # Reward for being in an open area

        return penalty

    def is_near_body(self, snake_head):
        """
        Check if the snake is near its own body (within a 1-step distance).
        """
        for segment in self.snake_coords[1:]:
            if calculate_distance(snake_head, segment) <= SPACE_SIZE:
                return True
        return False

    def calculate_open_space(self, snake_head):
        """
        Calculate the amount of open space available around the snake.
        Larger open space rewards the snake.
        """
        open_space = 0
        x, y = snake_head
        potential_moves = [(x + SPACE_SIZE, y), (x - SPACE_SIZE, y), (x, y + SPACE_SIZE), (x, y - SPACE_SIZE)]

        for move in potential_moves:
            if move not in self.snake_coords and self.is_within_bounds(move):
                open_space += 1  # Increase for every valid open space

        return open_space

    def is_within_bounds(self, pos):
        """
        Check if a position is within the game boundaries.
        """
        x, y = pos
        return 0 <= x < WIDTH and 0 <= y < HEIGHT

    def move_snake(self):
        """Moves the snake in the current direction"""
        x, y = self.snake_coords[0]

        if self.direction == 'up':
            y -= SPACE_SIZE
        elif self.direction == 'down':
            y += SPACE_SIZE
        elif self.direction == 'left':
            x -= SPACE_SIZE
        elif self.direction == 'right':
            x += SPACE_SIZE

        # Insert new head position
        self.snake_coords.insert(0, [x, y])
        new_square = self.create_square(x, y)
        self.snake_squares.insert(0, new_square)

    def remove_tail(self):
        """Removes the last part of the snake (the tail)"""
        tail_square = self.snake_squares.pop()
        self.canvas.delete(tail_square)
        del self.snake_coords[-1]

    def reset_game(self):
        """Resets the game to its initial state and starts the next episode."""
        self.canvas.delete('all')  # Clear the canvas
        self.score = 0  # Reset score
        self.label.config(text=f"Score: {self.score}")  # Update score label
        
        self.snake_coords = [[100, 100], [80, 100], [60, 100]]
        self.snake_squares = [self.create_square(x, y) for x, y in self.snake_coords]

        self.food_coord = self.place_food()
        self.food_square = self.create_oval(*self.food_coord)
        
        self.direction = 'right'
        self.window.after(SPEED, self.run_game)

    def save_checkpoint(self):
        # Save the model weights
        self.model.save_weights(SAVE_PATH)
        print(f"Checkpoint saved to {SAVE_PATH}")

        checkpoint_data = {
            'episode': self.episode,
            'score': self.score,
            'epsilon': EPSILON,
        }

        metadata_path = SAVE_PATH.replace('.h5', '_data.json')

        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                try:
                    all_data = json.load(f)

                    if not isinstance(all_data, list):
                        all_data = [all_data]
                except json.JSONDecodeError:
                    all_data = []
        else:
            all_data = []

        # Append new data
        all_data.append(checkpoint_data)

        # Save updated metadata
        with open(metadata_path, 'w') as f:
            json.dump(all_data, f, indent=4)

        print(f"Checkpoint metadata updated in {metadata_path}")

# The math
def calculate_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

# Start the game and train AI
game = SnakeGameAI()

game.window.protocol("WM_DELETE_WINDOW", game.save_checkpoint)
game.window.mainloop()