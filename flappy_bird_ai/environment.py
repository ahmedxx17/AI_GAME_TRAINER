"""
environment.py - Flappy Bird game environment built from scratch with pygame.

This module contains the FlappyBirdEnv class which implements the game logic,
physics, collision detection, and rendering. The AI agent interacts with this
environment by receiving state observations and sending actions.

No image files are used — all graphics are drawn with pygame primitives.
"""

import random
import numpy as np
import pygame


class FlappyBirdEnv:
    """
    A Flappy Bird game environment for reinforcement learning.

    The bird is fixed horizontally and moves only vertically.
    Pipes scroll from right to left. The agent's only choice
    each frame is: do nothing (0) or flap (1).
    """

    # ─── Game Constants ─────────────────────────────────────────────
    SCREEN_W    = 400       # Window width in pixels
    SCREEN_H    = 600       # Window height in pixels
    GRAVITY     = 0.5       # Downward acceleration per frame
    FLAP_STR    = -8        # Upward velocity on flap (negative = up in pygame)
    PIPE_SPEED  = 3         # Horizontal speed of pipes moving left
    PIPE_GAP    = 150       # Vertical gap between top and bottom pipes
    PIPE_WIDTH  = 60        # Width of each pipe in pixels
    BIRD_X      = 50        # Fixed horizontal position of the bird
    BIRD_RADIUS = 15        # Radius of the bird circle

    def __init__(self):
        """Initialize the environment with default values."""
        self.bird_y = 0.0       # Bird's vertical position
        self.bird_vel = 0.0     # Bird's vertical velocity
        self.pipe_x = 0.0       # Pipe's horizontal position
        self.gap_y = 0.0        # Center of the pipe gap
        self.score = 0          # Number of pipes successfully passed
        self.passed = False     # Flag to track if current pipe has been passed

    def reset(self):
        """
        Reset the environment to its initial state for a new episode.

        Returns:
            numpy.ndarray: Initial state vector of shape (4,), dtype float32.
        """
        self.bird_y = self.SCREEN_H / 2     # Start bird at vertical center
        self.bird_vel = 0.0                  # Zero initial velocity
        self.score = 0                       # Reset score
        self.passed = False                  # Reset pipe-passed flag

        # Spawn first pipe at the right edge with a random gap position
        self.pipe_x = self.SCREEN_W
        # Gap center is placed randomly, leaving margin so pipes are visible
        self.gap_y = random.randint(
            self.PIPE_GAP // 2 + 20,                      # Top margin
            self.SCREEN_H - self.PIPE_GAP // 2 - 20       # Bottom margin
        )

        return self._get_state()

    def step(self, action):
        """
        Advance the game by one frame given the agent's action.

        Parameters:
            action (int): 0 = do nothing, 1 = flap (apply upward velocity).

        Returns:
            tuple: (state, reward, done)
                - state (numpy.ndarray): New state vector of shape (4,).
                - reward (float): Reward received this frame.
                - done (bool): True if the bird crashed (episode over).
        """
        # ── Apply physics ───────────────────────────────────────────
        self.bird_vel += self.GRAVITY       # Gravity pulls the bird down each frame
        if action == 1:
            self.bird_vel = self.FLAP_STR   # Flap overrides velocity with upward impulse
        self.bird_y += self.bird_vel        # Update vertical position

        # ── Move pipes ──────────────────────────────────────────────
        self.pipe_x -= self.PIPE_SPEED      # Pipes scroll left

        # ── Respawn pipe when it goes off the left edge ─────────────
        if self.pipe_x + self.PIPE_WIDTH < 0:
            self.pipe_x = self.SCREEN_W     # New pipe spawns at right edge
            self.gap_y = random.randint(
                self.PIPE_GAP // 2 + 20,
                self.SCREEN_H - self.PIPE_GAP // 2 - 20
            )
            self.passed = False             # Reset passed flag for new pipe

        # ── Check if bird passed the pipe (for scoring) ─────────────
        # The bird passes the pipe when the pipe's right edge moves past the bird
        pipe_passed = False
        if not self.passed and self.pipe_x + self.PIPE_WIDTH < self.BIRD_X:
            self.score += 1
            self.passed = True
            pipe_passed = True

        # ── Collision detection ─────────────────────────────────────
        done = self._check_collision()

        # ── Determine reward ────────────────────────────────────────
        if done:
            reward = -1.0           # Penalty for crashing
        elif pipe_passed:
            reward = 1.0            # Bonus for passing a pipe
        else:
            reward = 0.1            # Small reward for staying alive

        return self._get_state(), reward, done

    def _check_collision(self):
        """
        Check if the bird has collided with a pipe, the ceiling, or the floor.

        Uses circle-vs-rectangle collision for the bird against pipe rectangles,
        and simple boundary checks for ceiling/floor.

        Returns:
            bool: True if a collision occurred, False otherwise.
        """
        # ── Ceiling and floor collision ─────────────────────────────
        if self.bird_y - self.BIRD_RADIUS <= 0:         # Hit ceiling
            return True
        if self.bird_y + self.BIRD_RADIUS >= self.SCREEN_H:  # Hit floor
            return True

        # ── Pipe collision (circle vs rectangle) ────────────────────
        # Calculate pipe boundaries
        pipe_top = self.gap_y - self.PIPE_GAP // 2      # Bottom edge of top pipe
        pipe_bottom = self.gap_y + self.PIPE_GAP // 2   # Top edge of bottom pipe

        # Check if bird is horizontally overlapping with the pipe
        if self.pipe_x < self.BIRD_X + self.BIRD_RADIUS and \
           self.pipe_x + self.PIPE_WIDTH > self.BIRD_X - self.BIRD_RADIUS:
            # Bird overlaps pipe horizontally — check vertical overlap
            if self.bird_y - self.BIRD_RADIUS < pipe_top:    # Hit top pipe
                return True
            if self.bird_y + self.BIRD_RADIUS > pipe_bottom:  # Hit bottom pipe
                return True

        return False

    def _get_state(self):
        """
        Build a normalized state vector for the neural network.

        Normalization is critical because it keeps all values in a similar range
        (approximately 0 to 1), preventing large numbers from dominating the
        network's learning and causing unstable gradients.

        Returns:
            numpy.ndarray: Normalized state of shape (4,), dtype float32.
                [0] bird_y / SCREEN_H            - Normalized bird height
                [1] (bird_vel + 10) / 20          - Normalized velocity (shifted to ~[0,1])
                [2] (pipe_x - BIRD_X) / SCREEN_W  - Normalized horizontal distance to pipe
                [3] (bird_y - gap_center) / SCREEN_H - Normalized vertical distance to gap
        """
        gap_center = self.gap_y     # Center of the pipe gap

        state = np.array([
            self.bird_y / self.SCREEN_H,                    # Bird's relative height
            (self.bird_vel + 10) / 20,                      # Normalized velocity
            (self.pipe_x - self.BIRD_X) / self.SCREEN_W,   # Distance to pipe
            (self.bird_y - gap_center) / self.SCREEN_H      # Offset from gap center
        ], dtype=np.float32)

        return state

    def render(self, screen):
        """
        Draw the current game state onto the provided pygame surface.

        All graphics use pygame.draw primitives — no image files are loaded.
        Note: This method does NOT call pygame.display.flip(); the caller
        is responsible for that.

        Parameters:
            screen (pygame.Surface): The pygame display surface to draw on.
        """
        # ── Background: dark navy ──────────────────────────────────
        screen.fill((15, 25, 40))

        # ── Calculate pipe positions ────────────────────────────────
        pipe_top = self.gap_y - self.PIPE_GAP // 2      # Bottom of top pipe
        pipe_bottom = self.gap_y + self.PIPE_GAP // 2   # Top of bottom pipe

        # ── Draw top pipe (green rectangle from y=0 down to pipe_top) ──
        top_pipe_rect = pygame.Rect(
            int(self.pipe_x), 0,
            self.PIPE_WIDTH, pipe_top
        )
        pygame.draw.rect(screen, (50, 180, 50), top_pipe_rect)         # Fill
        pygame.draw.rect(screen, (30, 120, 30), top_pipe_rect, 2)      # Border

        # ── Draw bottom pipe (green rectangle from pipe_bottom to screen bottom) ──
        bottom_pipe_rect = pygame.Rect(
            int(self.pipe_x), pipe_bottom,
            self.PIPE_WIDTH, self.SCREEN_H - pipe_bottom
        )
        pygame.draw.rect(screen, (50, 180, 50), bottom_pipe_rect)      # Fill
        pygame.draw.rect(screen, (30, 120, 30), bottom_pipe_rect, 2)   # Border

        # ── Draw bird (yellow circle) ───────────────────────────────
        pygame.draw.circle(
            screen,
            (255, 220, 50),                         # Yellow color
            (self.BIRD_X, int(self.bird_y)),         # Center position
            self.BIRD_RADIUS                         # Radius
        )

        # ── Draw score text (white, top-left) ──────────────────────
        font = pygame.font.SysFont(None, 24)
        score_surface = font.render(f"Score: {self.score}", True, (255, 255, 255))
        screen.blit(score_surface, (10, 10))
