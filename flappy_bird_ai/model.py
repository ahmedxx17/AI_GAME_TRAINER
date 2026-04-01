"""model.py - Deep Q-Network (DQN) neural network architecture."""

import torch
import torch.nn as nn

from config import ACTION_SIZE, HIDDEN_SIZE, STATE_SIZE


class DQN(nn.Module):
    """
    Deep Q-Network with two hidden layers.

    Architecture:
        Input (STATE_SIZE) -> Linear(HIDDEN_SIZE) -> ReLU ->
        Linear(HIDDEN_SIZE) -> ReLU -> Linear(ACTION_SIZE) -> Output

    The output layer has NO activation function (no softmax or sigmoid)
    because Q-values can be any real number — they represent expected
    cumulative rewards, which are not bounded to [0, 1].
    """

    def __init__(
        self,
        state_size: int = STATE_SIZE,
        action_size: int = ACTION_SIZE,
        hidden_size: int = HIDDEN_SIZE,
    ) -> None:
        """
        Initialize the DQN with a sequential network.

        Layer breakdown:
            - Layer 1: state_size inputs -> hidden_size neurons
            - Layer 2: hidden_size neurons -> hidden_size neurons
            - Layer 3: hidden_size neurons -> action_size outputs
        """
        super(DQN, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),      # Input layer: state -> hidden units
            nn.ReLU(),             # Activation: introduces non-linearity
            nn.Linear(hidden_size, hidden_size),     # Hidden layer: deeper pattern recognition
            nn.ReLU(),             # Activation
            nn.Linear(hidden_size, action_size)      # Output layer: Q-value per action
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Parameters:
            x (torch.Tensor): Input state tensor of shape (batch_size, state_size).

        Returns:
            torch.Tensor: Q-values of shape (batch_size, action_size).
        """
        return self.network(x)      # Raw Q-values, no activation on output
