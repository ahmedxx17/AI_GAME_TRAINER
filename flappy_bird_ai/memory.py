"""
memory.py - Experience Replay Memory for DQN training.

This module implements a fixed-size replay buffer that stores past experiences
(state, action, reward, next_state, done) and allows random sampling.

Why replay memory is important:
    1. Breaks temporal correlation: Consecutive game frames are highly correlated.
       Training on them in order makes the network overfit to recent patterns.
       Random sampling from memory gives diverse, uncorrelated training batches.

    2. Data efficiency: Each experience can be used many times for training,
       not just once when it happens.

    3. Smoother learning: The random sampling averages out the noise from
       individual experiences, leading to more stable gradient updates.
"""

import random
from collections import deque


class ReplayMemory:
    """
    A fixed-size circular buffer that stores experience tuples for DQN training.

    When the buffer is full, the oldest experience is automatically discarded
    to make room for the newest one (FIFO behavior via deque).
    """

    def __init__(self, capacity=10000):
        """
        Initialize the replay memory.

        Parameters:
            capacity (int): Maximum number of experiences to store.
                           Default is 10000, matching config.MEMORY_CAPACITY.
        """
        # deque with maxlen automatically drops oldest items when full
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """
        Store a single experience in the replay buffer.

        Parameters:
            state (numpy.ndarray): The state observed before taking the action.
            action (int): The action taken (0 = do nothing, 1 = flap).
            reward (float): The reward received after taking the action.
            next_state (numpy.ndarray): The state observed after taking the action.
            done (float): 1.0 if the episode ended, 0.0 otherwise.
        """
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        Randomly sample a batch of experiences from the buffer.

        Random sampling is the key mechanism that breaks temporal correlation
        between training examples.

        Parameters:
            batch_size (int): Number of experiences to sample.

        Returns:
            list: A list of (state, action, reward, next_state, done) tuples.
        """
        return random.sample(self.memory, batch_size)

    def __len__(self):
        """
        Return the current number of experiences stored.

        Returns:
            int: Number of experiences currently in the buffer.
        """
        return len(self.memory)
