"""
environment.py - Snake game environment with pixel-based state output.

The environment renders each frame as an 84×84 grayscale image so that a
convolutional neural network can learn directly from raw visual input,
mirroring the approach used in the DeepMind Atari DQN paper.
"""

import random
from typing import List, Tuple

import numpy as np

from config import (
    DISPLAY_SIZE,
    FRAME_SIZE,
    GRID_SIZE,
    HUNGER_LIMIT,
    REWARD_CLOSER,
    REWARD_DEATH,
    REWARD_FARTHER,
    REWARD_FOOD,
)

# ─── Direction vectors (dx, dy) ────────────────────────────────────
UP    = ( 0, -1)
RIGHT = ( 1,  0)
DOWN  = ( 0,  1)
LEFT  = (-1,  0)

DIRECTIONS = [UP, RIGHT, DOWN, LEFT]
OPPOSITES  = {UP: DOWN, DOWN: UP, LEFT: RIGHT, RIGHT: LEFT}


class SnakeEnv:
    """Snake game that outputs raw pixel frames for a CNN-based agent."""

    def __init__(self) -> None:
        self.grid_size = GRID_SIZE
        self.snake: List[Tuple[int, int]] = []
        self.direction = RIGHT
        self.food: Tuple[int, int] = (0, 0)
        self.score = 0
        self.done = False
        self._steps_hungry = 0
        self._font = None

    # ────────────────────────────────────────────────────────────────
    #  Core RL interface
    # ────────────────────────────────────────────────────────────────

    def reset(self) -> np.ndarray:
        """Reset environment for a new episode and return the first frame."""
        center = self.grid_size // 2
        self.snake = [
            (center, center),
            (center - 1, center),
            (center - 2, center),
        ]
        self.direction = RIGHT
        self.food = self._place_food()
        self.score = 0
        self.done = False
        self._steps_hungry = 0
        return self.get_frame()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """
        Advance the game by one frame.

        Parameters:
            action: 0 = up, 1 = right, 2 = down, 3 = left.

        Returns:
            (frame, reward, done) tuple.
        """
        # ── Change direction (reversing is not allowed) ─────────────
        new_dir = DIRECTIONS[action]
        if new_dir != OPPOSITES.get(self.direction):
            self.direction = new_dir

        old_dist = self._manhattan_distance()

        # ── Move the snake ──────────────────────────────────────────
        head_x, head_y = self.snake[0]
        dx, dy = self.direction
        new_head = (head_x + dx, head_y + dy)

        # ── Collision check ─────────────────────────────────────────
        if self._is_collision(new_head):
            self.done = True
            return self.get_frame(), REWARD_DEATH, True

        self.snake.insert(0, new_head)
        self._steps_hungry += 1

        # ── Food check ──────────────────────────────────────────────
        if new_head == self.food:
            self.score += 1
            self._steps_hungry = 0
            reward = REWARD_FOOD
            if len(self.snake) < self.grid_size * self.grid_size:
                self.food = self._place_food()
            else:
                self.done = True
                return self.get_frame(), reward, True
        else:
            self.snake.pop()
            new_dist = self._manhattan_distance()
            reward = REWARD_CLOSER if new_dist < old_dist else REWARD_FARTHER

        # ── Hunger timeout ──────────────────────────────────────────
        if self._steps_hungry >= HUNGER_LIMIT:
            self.done = True

        return self.get_frame(), reward, self.done

    # ────────────────────────────────────────────────────────────────
    #  Frame rendering (for CNN input)
    # ────────────────────────────────────────────────────────────────

    def get_frame(self) -> np.ndarray:
        """
        Render the current game state as an 84×84 grayscale image.

        Pixel values:
            0   — empty cell (black)
            85  — food
            170 — snake body
            255 — snake head (white)
        """
        grid = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)

        fx, fy = self.food
        grid[fy, fx] = 85

        for x, y in self.snake[1:]:
            grid[y, x] = 170

        hx, hy = self.snake[0]
        if 0 <= hy < self.grid_size and 0 <= hx < self.grid_size:
            grid[hy, hx] = 255

        return _resize_nearest(grid, FRAME_SIZE, FRAME_SIZE)

    # ────────────────────────────────────────────────────────────────
    #  Pygame rendering (for display / human play)
    # ────────────────────────────────────────────────────────────────

    def render(self, screen) -> None:  # noqa: ANN001
        """Draw the game on a pygame surface for human viewing."""
        import pygame

        cell = DISPLAY_SIZE // self.grid_size

        # Background
        screen.fill((20, 20, 20))

        # Grid lines
        for i in range(self.grid_size + 1):
            pygame.draw.line(
                screen, (40, 40, 40), (i * cell, 0), (i * cell, DISPLAY_SIZE)
            )
            pygame.draw.line(
                screen, (40, 40, 40), (0, i * cell), (DISPLAY_SIZE, i * cell)
            )

        # Food
        fx, fy = self.food
        pygame.draw.rect(
            screen,
            (220, 50, 50),
            (fx * cell + 2, fy * cell + 2, cell - 4, cell - 4),
            border_radius=4,
        )

        # Snake
        for i, (x, y) in enumerate(self.snake):
            colour = (0, 230, 0) if i == 0 else (0, 180, 0)
            pygame.draw.rect(
                screen,
                colour,
                (x * cell + 1, y * cell + 1, cell - 2, cell - 2),
                border_radius=3,
            )

        # Score overlay
        if self._font is None:
            self._font = pygame.font.SysFont(None, 30)
        text = self._font.render(f"Score: {self.score}", True, (255, 255, 255))
        screen.blit(text, (10, 10))

    # ────────────────────────────────────────────────────────────────
    #  Helpers
    # ────────────────────────────────────────────────────────────────

    def _place_food(self) -> Tuple[int, int]:
        empty = [
            (x, y)
            for x in range(self.grid_size)
            for y in range(self.grid_size)
            if (x, y) not in self.snake
        ]
        return random.choice(empty) if empty else self.snake[-1]

    def _manhattan_distance(self) -> int:
        hx, hy = self.snake[0]
        fx, fy = self.food
        return abs(hx - fx) + abs(hy - fy)

    def _is_collision(self, pos: Tuple[int, int]) -> bool:
        x, y = pos
        if x < 0 or x >= self.grid_size or y < 0 or y >= self.grid_size:
            return True
        return pos in self.snake[:-1]


# ─── Utility ────────────────────────────────────────────────────────

def _resize_nearest(arr: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Up-scale a 2-D array using nearest-neighbour interpolation."""
    old_h, old_w = arr.shape
    rows = (np.arange(target_h) * old_h / target_h).astype(int)
    cols = (np.arange(target_w) * old_w / target_w).astype(int)
    return arr[rows][:, cols]
