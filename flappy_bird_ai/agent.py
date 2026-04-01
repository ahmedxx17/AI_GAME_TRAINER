"""
agent.py - DQN Agent that learns to play Flappy Bird.

This module contains the DQNAgent class — the AI "brain" that:
    1. Chooses actions using an epsilon-greedy policy
    2. Stores experiences in replay memory
    3. Trains the neural network using the Bellman equation

The agent starts by exploring randomly (high epsilon) and gradually
shifts to exploiting its learned knowledge (low epsilon).
"""

import random
import numpy as np
import torch
import torch.nn as nn

from model import DQN
from memory import ReplayMemory
from config import (
    BATCH_SIZE, LEARNING_RATE, GAMMA,
    EPSILON_START, EPSILON_MIN, EPSILON_DECAY,
    MEMORY_CAPACITY
)


class DQNAgent:
    """
    Deep Q-Learning agent that learns to play Flappy Bird through trial and error.

    The agent uses:
        - Epsilon-greedy exploration to balance exploration vs exploitation
        - Experience replay to break temporal correlations in training data
        - The Bellman equation to compute target Q-values for learning
    """

    def __init__(self):
        """
        Initialize the DQN agent with network, memory, optimizer, and exploration settings.
        """
        # ── Device selection: use GPU if available for faster training ──
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # ── Neural network: estimates Q-values for each action ──────────
        self.model = DQN().to(self.device)

        # ── Replay memory: stores past experiences for training ─────────
        self.memory = ReplayMemory(capacity=MEMORY_CAPACITY)

        # ── Optimizer: Adam is widely used for its adaptive learning rate ──
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)

        # ── Loss function: MSE between predicted Q and target Q ─────────
        self.criterion = nn.MSELoss()

        # ── Exploration rate: starts high (explore) and decays (exploit) ─
        self.epsilon = EPSILON_START

    def choose_action(self, state):
        """
        Select an action using epsilon-greedy policy.

        With probability epsilon: choose a random action (exploration).
        With probability 1-epsilon: choose the action with highest Q-value (exploitation).

        Exploration is essential early in training because the network's Q-value
        estimates are essentially random — the agent needs to try different things
        to discover what works.

        Parameters:
            state (numpy.ndarray): Current state vector of shape (4,).

        Returns:
            int: Action to take (0 = do nothing, 1 = flap).
        """
        # ── Explore: random action ─────────────────────────────────
        if random.random() < self.epsilon:
            return random.randint(0, 1)

        # ── Exploit: use the network to pick the best action ───────
        # Convert numpy state to PyTorch tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        # no_grad() disables gradient computation since we're only doing inference,
        # not training. This saves memory and computation.
        with torch.no_grad():
            q_values = self.model(state_tensor)     # Get Q-values for both actions

        # argmax returns the index of the highest Q-value (best action)
        return q_values.argmax().item()

    def train(self):
        """
        Perform one training step using a batch sampled from replay memory.

        This implements the core DQN learning algorithm:
            1. Sample a random batch of past experiences
            2. Compute current Q-values (what the network currently predicts)
            3. Compute target Q-values using the Bellman equation:
                   target = reward + gamma * max(Q(next_state)) * (1 - done)
            4. Minimize the difference between current and target Q-values

        The Bellman equation says: the Q-value of a state-action pair should equal
        the immediate reward plus the discounted value of the best action in the
        next state. If the episode ended (done=True), there is no next state,
        so the target is just the immediate reward.
        """
        # Don't train until we have enough experiences for a full batch
        if len(self.memory) < BATCH_SIZE:
            return

        # ── Sample a random batch from replay memory ────────────────
        batch = self.memory.sample(BATCH_SIZE)

        # ── Unpack the batch into separate arrays ───────────────────
        # zip(*batch) transposes the list of tuples into tuples of lists
        states, actions, rewards, next_states, dones = zip(*batch)

        # ── Convert everything to PyTorch tensors on the correct device ──
        states      = torch.FloatTensor(np.array(states)).to(self.device)
        actions     = torch.LongTensor(actions).to(self.device)
        rewards     = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones       = torch.FloatTensor(dones).to(self.device)

        # ── Compute current Q-values ────────────────────────────────
        # model(states) returns Q-values for ALL actions: shape (batch, 2)
        # .gather(1, actions) selects the Q-value for the action that was actually taken
        # .squeeze() removes the extra dimension, giving shape (batch,)
        current_q = self.model(states).gather(1, actions.unsqueeze(1)).squeeze()

        # ── Compute target Q-values using the Bellman equation ──────
        # no_grad() because targets should be treated as fixed values,
        # not part of the computation graph (we don't backprop through targets)
        with torch.no_grad():
            # max(1)[0] gets the maximum Q-value across actions for each next_state
            max_next_q = self.model(next_states).max(1)[0]

            # Bellman equation: target = reward + gamma * max_next_q * (1 - done)
            # (1 - done) zeroes out future value when the episode ended
            target_q = rewards + (1 - dones) * GAMMA * max_next_q

        # ── Compute loss and update the network ─────────────────────
        loss = self.criterion(current_q, target_q)  # MSE between predicted and target

        self.optimizer.zero_grad()  # Clear old gradients (PyTorch accumulates by default)
        loss.backward()             # Compute gradients via backpropagation
        self.optimizer.step()       # Update network weights using Adam optimizer

    def decay_epsilon(self):
        """
        Reduce the exploration rate after each episode.

        Epsilon decays multiplicatively: epsilon *= EPSILON_DECAY
        This gradually shifts the agent from exploring (random actions)
        to exploiting (using learned Q-values).

        The decay is clamped at EPSILON_MIN so the agent always retains
        a small amount of exploration (1%) to handle unexpected situations.
        """
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)
