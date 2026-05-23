"""
memory.py - Experience replay buffer for the CNN-DQN agent.

Each transition stores full frame-stacked states as uint8 arrays to keep
memory usage low (~28 KB per transition for 4 × 84 × 84 uint8 frames).
"""

import random
from collections import deque
from typing import Deque, List, Tuple

import numpy as np

# (stacked_frames, action, reward, next_stacked_frames, done)
Experience = Tuple[np.ndarray, int, float, np.ndarray, float]


class ReplayMemory:
    """Fixed-size circular buffer of frame-stacked experience tuples."""

    def __init__(self, capacity: int) -> None:
        self.memory: Deque[Experience] = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: float,
    ) -> None:
        """Store one (state, action, reward, next_state, done) transition."""
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> List[Experience]:
        """Randomly sample a batch of transitions."""
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        return len(self.memory)
