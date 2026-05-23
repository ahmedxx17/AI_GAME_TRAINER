"""
agent.py - Vision-based DQN agent with frame stacking.

The agent maintains a sliding window of the last FRAME_STACK frames
(default 4) and feeds them as channels to a convolutional neural network.
Frame stacking gives the CNN temporal information — for example, the
direction the snake is moving — which a single static frame cannot convey.
"""

import random
from collections import deque
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from config import (
    ACTION_SIZE,
    BATCH_SIZE,
    EPSILON_DECAY,
    EPSILON_MIN,
    EPSILON_START,
    FRAME_STACK,
    GAMMA,
    LEARNING_RATE,
    MEMORY_CAPACITY,
    TARGET_UPDATE_FREQ,
)
from memory import ReplayMemory
from model import CNNDQN


class CNNAgent:
    """DQN agent that learns from raw pixel frames via a CNN."""

    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # ── Policy and target networks ──────────────────────────────
        self.model = CNNDQN().to(self.device)
        self.target_model = CNNDQN().to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        # ── Replay memory ───────────────────────────────────────────
        self.memory = ReplayMemory(capacity=MEMORY_CAPACITY)

        # ── Optimiser and loss ──────────────────────────────────────
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.criterion = nn.SmoothL1Loss()

        # ── Exploration ─────────────────────────────────────────────
        self.epsilon = EPSILON_START

        # ── Internal counters ───────────────────────────────────────
        self.train_step_count = 0

        # ── Frame stack (sliding window of recent frames) ───────────
        self.frames: deque[np.ndarray] = deque(maxlen=FRAME_STACK)

    # ────────────────────────────────────────────────────────────────
    #  Frame management
    # ────────────────────────────────────────────────────────────────

    def reset_frames(self, frame: np.ndarray) -> None:
        """Fill the frame stack with copies of the initial frame."""
        self.frames.clear()
        for _ in range(FRAME_STACK):
            self.frames.append(frame)

    def push_frame(self, frame: np.ndarray) -> None:
        """Append a new frame; the oldest frame is dropped automatically."""
        self.frames.append(frame)

    def get_state(self) -> np.ndarray:
        """Return the current stacked state as (FRAME_STACK, 84, 84) uint8."""
        return np.array(self.frames, dtype=np.uint8)

    # ────────────────────────────────────────────────────────────────
    #  Action selection
    # ────────────────────────────────────────────────────────────────

    def choose_action(self, state: np.ndarray) -> int:
        """
        Epsilon-greedy action selection.

        Parameters:
            state: Stacked frames of shape (FRAME_STACK, 84, 84).

        Returns:
            Action index (0 = up, 1 = right, 2 = down, 3 = left).
        """
        if random.random() < self.epsilon:
            return random.randint(0, ACTION_SIZE - 1)

        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state_t)
        return q_values.argmax().item()

    # ────────────────────────────────────────────────────────────────
    #  Training
    # ────────────────────────────────────────────────────────────────

    def train(self) -> Optional[float]:
        """
        One gradient step on a mini-batch from replay memory.

        Returns the loss value, or None if the buffer is too small.
        """
        if len(self.memory) < BATCH_SIZE:
            return None

        batch = self.memory.sample(BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)

        states      = torch.FloatTensor(np.array(states)).to(self.device)
        actions     = torch.LongTensor(actions).to(self.device)
        rewards     = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones       = torch.FloatTensor(dones).to(self.device)

        # Current Q-values for the actions that were actually taken
        current_q = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q-values via Bellman equation (computed with frozen target net)
        with torch.no_grad():
            max_next_q = self.target_model(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * GAMMA * max_next_q

        loss = self.criterion(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        # ── Periodically sync target network ────────────────────────
        self.train_step_count += 1
        if self.train_step_count % TARGET_UPDATE_FREQ == 0:
            self.target_model.load_state_dict(self.model.state_dict())

        return loss.item()

    # ────────────────────────────────────────────────────────────────
    #  Exploration decay
    # ────────────────────────────────────────────────────────────────

    def decay_epsilon(self) -> None:
        """Multiplicative epsilon decay (called once per episode)."""
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)
