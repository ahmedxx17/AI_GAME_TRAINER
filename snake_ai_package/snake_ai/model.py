"""
model.py - Convolutional Deep Q-Network for vision-based game playing.

This CNN architecture is inspired by the DeepMind Atari DQN paper
(Mnih et al., 2015).  It takes a stack of raw grayscale game frames
and outputs a Q-value for every possible action.

Architecture:
    Input  (FRAME_STACK × 84 × 84)
      → Conv2d(16, 8×8, stride 4) → ReLU      — coarse spatial features
      → Conv2d(32, 4×4, stride 2) → ReLU      — mid-level features
      → Conv2d(32, 3×3, stride 1) → ReLU      — fine-grained features
      → Flatten (32 × 7 × 7 = 1 568)
      → Linear(256) → ReLU
      → Linear(ACTION_SIZE)                    — one Q-value per action
"""

import torch
import torch.nn as nn

from config import ACTION_SIZE, FRAME_STACK


class CNNDQN(nn.Module):
    """Convolutional DQN that processes raw pixel frames."""

    def __init__(self) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(FRAME_STACK, 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        self.head = nn.Sequential(
            nn.Linear(32 * 7 * 7, 256),
            nn.ReLU(),
            nn.Linear(256, ACTION_SIZE),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters:
            x: Batch of stacked frames, shape (B, FRAME_STACK, 84, 84),
               with pixel values in [0, 255].

        Returns:
            Q-values of shape (B, ACTION_SIZE).
        """
        x = x / 255.0
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.head(x)
