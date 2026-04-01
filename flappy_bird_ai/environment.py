"""environment.py - Flappy Bird game environment built with pygame."""

import random
from typing import List, Tuple, TypedDict

import numpy as np
import pygame

from config import (
    MULTI_PIPE,
    NUM_PIPES,
    PIPE_SPACING,
    REWARD_ALIVE,
    REWARD_DEATH,
    REWARD_PASS,
)


class Pipe(TypedDict):
    x: float
    gap_y: int
    passed: bool


class FlappyBirdEnv:
    """A Flappy Bird game environment for reinforcement learning."""

    SCREEN_W = 400
    SCREEN_H = 600
    GRAVITY = 0.5
    FLAP_STR = -8
    PIPE_SPEED = 3
    PIPE_GAP = 150
    PIPE_WIDTH = 60
    BIRD_X = 50
    BIRD_RADIUS = 15
    GAP_MARGIN = 20

    def __init__(self) -> None:
        self.bird_y = 0.0
        self.bird_vel = 0.0
        self.score = 0
        self.pipes: List[Pipe] = []
        self._font: pygame.font.Font | None = None

    def reset(self) -> np.ndarray:
        """Reset environment state for a new episode."""
        self.bird_y = self.SCREEN_H / 2
        self.bird_vel = 0.0
        self.score = 0

        pipe_count = NUM_PIPES if MULTI_PIPE else 1
        self.pipes = []
        for i in range(pipe_count):
            self.pipes.append(
                Pipe(
                    x=float(self.SCREEN_W + i * PIPE_SPACING),
                    gap_y=self._random_gap_y(),
                    passed=False,
                )
            )

        return self._get_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """Advance the game by one frame."""
        self.bird_vel += self.GRAVITY
        if action == 1:
            self.bird_vel = self.FLAP_STR
        self.bird_y += self.bird_vel

        for pipe in self.pipes:
            pipe["x"] -= self.PIPE_SPEED

        self._respawn_offscreen_pipes()

        pipe_passed = False
        for pipe in self.pipes:
            if (not pipe["passed"]) and (pipe["x"] + self.PIPE_WIDTH < self.BIRD_X):
                pipe["passed"] = True
                self.score += 1
                pipe_passed = True

        done = self._check_collision()

        if done:
            reward = REWARD_DEATH
        elif pipe_passed:
            reward = REWARD_PASS
        else:
            reward = REWARD_ALIVE

        return self._get_state(), reward, done

    def _respawn_offscreen_pipes(self) -> None:
        """Respawn pipes that moved fully offscreen to keep flow continuous."""
        if not self.pipes:
            return

        for pipe in self.pipes:
            if pipe["x"] + self.PIPE_WIDTH < 0:
                max_x = max(existing_pipe["x"] for existing_pipe in self.pipes)
                pipe["x"] = max_x + PIPE_SPACING
                pipe["gap_y"] = self._random_gap_y()
                pipe["passed"] = False

    def _random_gap_y(self) -> int:
        """Create a valid random gap center with top/bottom margin."""
        return random.randint(
            self.PIPE_GAP // 2 + self.GAP_MARGIN,
            self.SCREEN_H - self.PIPE_GAP // 2 - self.GAP_MARGIN,
        )

    def _nearest_pipe(self) -> Pipe:
        """Get the nearest relevant pipe to the bird."""
        ahead_pipes = [
            pipe
            for pipe in self.pipes
            if pipe["x"] + self.PIPE_WIDTH >= self.BIRD_X - self.BIRD_RADIUS
        ]

        if ahead_pipes:
            return min(ahead_pipes, key=lambda pipe: pipe["x"])

        return min(self.pipes, key=lambda pipe: pipe["x"])

    def _check_collision(self) -> bool:
        """Check if bird hit boundaries or any pipe."""
        if self.bird_y - self.BIRD_RADIUS <= 0:
            return True
        if self.bird_y + self.BIRD_RADIUS >= self.SCREEN_H:
            return True

        for pipe in self.pipes:
            pipe_x = pipe["x"]
            gap_y = pipe["gap_y"]
            pipe_top = gap_y - self.PIPE_GAP // 2
            pipe_bottom = gap_y + self.PIPE_GAP // 2

            if pipe_x < self.BIRD_X + self.BIRD_RADIUS and pipe_x + self.PIPE_WIDTH > self.BIRD_X - self.BIRD_RADIUS:
                if self.bird_y - self.BIRD_RADIUS < pipe_top:
                    return True
                if self.bird_y + self.BIRD_RADIUS > pipe_bottom:
                    return True

        return False

    def _get_state(self) -> np.ndarray:
        """Build a normalized 6D state vector for the neural network."""
        pipe = self._nearest_pipe()
        pipe_x = pipe["x"]
        gap_y = pipe["gap_y"]

        pipe_top = gap_y - self.PIPE_GAP // 2
        pipe_bottom = gap_y + self.PIPE_GAP // 2

        state = np.array(
            [
                self.bird_y / self.SCREEN_H,
                (self.bird_vel + 10) / 20,
                (pipe_x - self.BIRD_X) / self.SCREEN_W,
                (self.bird_y - gap_y) / self.SCREEN_H,
                (self.bird_y - pipe_top) / self.SCREEN_H,
                (pipe_bottom - self.bird_y) / self.SCREEN_H,
            ],
            dtype=np.float32,
        )
        return state

    def render(self, screen: pygame.Surface) -> None:
        """Draw current game state onto the provided pygame surface."""
        screen.fill((15, 25, 40))

        for pipe in self.pipes:
            pipe_x = int(pipe["x"])
            gap_y = pipe["gap_y"]
            pipe_top = gap_y - self.PIPE_GAP // 2
            pipe_bottom = gap_y + self.PIPE_GAP // 2

            top_pipe_rect = pygame.Rect(pipe_x, 0, self.PIPE_WIDTH, pipe_top)
            pygame.draw.rect(screen, (50, 180, 50), top_pipe_rect)
            pygame.draw.rect(screen, (30, 120, 30), top_pipe_rect, 2)

            bottom_pipe_rect = pygame.Rect(
                pipe_x,
                pipe_bottom,
                self.PIPE_WIDTH,
                self.SCREEN_H - pipe_bottom,
            )
            pygame.draw.rect(screen, (50, 180, 50), bottom_pipe_rect)
            pygame.draw.rect(screen, (30, 120, 30), bottom_pipe_rect, 2)

        pygame.draw.circle(
            screen,
            (255, 220, 50),
            (self.BIRD_X, int(self.bird_y)),
            self.BIRD_RADIUS,
        )

        if self._font is None:
            self._font = pygame.font.SysFont(None, 24)
        score_surface = self._font.render(f"Score: {self.score}", True, (255, 255, 255))
        screen.blit(score_surface, (10, 10))
