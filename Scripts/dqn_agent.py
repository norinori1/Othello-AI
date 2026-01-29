"""
DQN (Deep Q-Network) Agent for Othello.
Implements a reinforcement learning agent using deep Q-learning.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from .board import BLACK, WHITE


class DQNetwork(nn.Module):
    """Deep Q-Network for Othello."""
    
    def __init__(self):
        """Initialize the network architecture as per specification."""
        super(DQNetwork, self).__init__()
        
        # Input: 8x8x3 (self pieces, opponent pieces, legal moves)
        # Conv layers as per spec
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 65)  # 64 board positions + 1 pass action
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch, 3, 8, 8)
            
        Returns:
            Q-values for all 65 actions
        """
        # Conv layers with ReLU
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x


class ReplayBuffer:
    """Experience replay buffer for DQN."""
    
    def __init__(self, capacity=10000):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum number of experiences to store
        """
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """
        Add an experience to the buffer.
        
        Args:
            state: Current state (8x8x3 numpy array)
            action: Action taken (row, col) or None
            reward: Reward received
            next_state: Next state (8x8x3 numpy array)
            done: Whether episode is done
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """
        Sample a batch of experiences.
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            Batch of experiences
        """
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """DQN Agent for playing Othello."""
    
    def __init__(self, color, learning_rate=0.001, gamma=0.99, epsilon=1.0, 
                 epsilon_min=0.1, epsilon_decay=0.995, device=None):
        """
        Initialize the DQN agent.
        
        Args:
            color: BLACK or WHITE
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            epsilon: Initial exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Epsilon decay rate per episode
            device: torch device (cpu/cuda)
        """
        self.color = color
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        # Create Q-networks
        self.policy_net = DQNetwork().to(self.device)
        self.target_net = DQNetwork().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=10000)
        
        # Training stats
        self.training_step = 0
        self.episode_count = 0
    
    def action_to_index(self, action):
        """
        Convert action (row, col) to index 0-64.
        Pass action (None) -> index 64
        
        Args:
            action: (row, col) or None
            
        Returns:
            int: Action index
        """
        if action is None:
            return 64
        row, col = action
        return row * 8 + col
    
    def index_to_action(self, index):
        """
        Convert index to action.
        
        Args:
            index: Action index 0-64
            
        Returns:
            tuple: (row, col) or None for pass
        """
        if index == 64:
            return None
        row = index // 8
        col = index % 8
        return (row, col)
    
    def select_move(self, board, training=False):
        """
        Select a move using epsilon-greedy policy.
        
        Args:
            board: Board object
            training: Whether in training mode
            
        Returns:
            tuple: (row, col) or None
        """
        valid_moves = board.valid_moves(self.color)
        
        if not valid_moves:
            return None
        
        # Get valid action indices
        valid_actions = [self.action_to_index(move) for move in valid_moves.keys()]
        
        # Epsilon-greedy exploration
        if training and random.random() < self.epsilon:
            # Random exploration
            action_idx = random.choice(valid_actions)
            return self.index_to_action(action_idx)
        
        # Greedy exploitation
        state = board.to_tensor(self.color)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_values = self.policy_net(state_tensor).squeeze(0)
        
        # Mask invalid actions (create directly on device)
        mask = torch.full((65,), float('-inf'), device=self.device)
        for idx in valid_actions:
            mask[idx] = 0
        q_values = q_values + mask
        
        # Select best valid action
        action_idx = q_values.argmax().item()
        return self.index_to_action(action_idx)
    
    def update_target_network(self):
        """Update target network with policy network weights."""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def train_step(self, batch_size=32):
        """
        Perform one training step.
        
        Args:
            batch_size: Batch size for training
            
        Returns:
            float: Loss value
        """
        if len(self.replay_buffer) < batch_size:
            return 0.0
        
        # Sample batch
        batch = self.replay_buffer.sample(batch_size)
        
        # Prepare batch tensors
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        
        for state, action, reward, next_state, done in batch:
            states.append(state)
            actions.append(self.action_to_index(action))
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
        
        # Convert to numpy arrays first, then to tensors
        states = torch.FloatTensor(np.stack(states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
        next_states = torch.FloatTensor(np.stack(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).to(self.device)
        
        # Compute Q(s, a)
        q_values = self.policy_net(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute target Q values
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss
        loss = self.loss_fn(q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.training_step += 1
        
        return loss.item()
    
    def decay_epsilon(self):
        """Decay epsilon after each episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.episode_count += 1
    
    def save(self, filepath):
        """
        Save the agent's model.
        
        Args:
            filepath: Path to save the model
        """
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_step': self.training_step,
            'episode_count': self.episode_count,
        }, filepath)
    
    def load(self, filepath):
        """
        Load the agent's model.
        
        Args:
            filepath: Path to load the model from
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.training_step = checkpoint['training_step']
        self.episode_count = checkpoint['episode_count']
