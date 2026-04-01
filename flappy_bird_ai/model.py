"""
model.py - Deep Q-Network (DQN) neural network architecture.

This module defines the neural network that the AI agent uses to estimate
Q-values for each possible action. The network takes a 4-dimensional state
vector as input and outputs 2 Q-values (one for each action: do nothing or flap).

The Q-value represents the expected total future reward for taking an action
in a given state. Higher Q-value = better action according to the network.
"""

import torch
import torch.nn as nn


class DQN(nn.Module):
    """
    Deep Q-Network with two hidden layers.

    Architecture:
        Input (4) -> Linear(64) -> ReLU -> Linear(64) -> ReLU -> Linear(2) -> Output

    The output layer has NO activation function (no softmax or sigmoid)
    because Q-values can be any real number — they represent expected
    cumulative rewards, which are not bounded to [0, 1].
    """

    def __init__(self):
        """
        Initialize the DQN with a sequential network.

        Layer breakdown:
            - Layer 1: 4 inputs (state features) -> 64 neurons
            - Layer 2: 64 neurons -> 64 neurons (deeper feature extraction)
            - Layer 3: 64 neurons -> 2 outputs (Q-value per action)
        """
        super(DQN, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(4, 64),      # Input layer: 4 state values -> 64 hidden units
            nn.ReLU(),             # Activation: introduces non-linearity
            nn.Linear(64, 64),     # Hidden layer: deeper pattern recognition
            nn.ReLU(),             # Activation
            nn.Linear(64, 2)       # Output layer: Q-value for each of 2 actions
        )

    def forward(self, x):
        """
        Forward pass through the network.

        Parameters:
            x (torch.Tensor): Input state tensor of shape (batch_size, 4).

        Returns:
            torch.Tensor: Q-values of shape (batch_size, 2).
                          Index 0 = Q-value for 'do nothing'
                          Index 1 = Q-value for 'flap'
        """
        return self.network(x)      # Raw Q-values, no activation on output
