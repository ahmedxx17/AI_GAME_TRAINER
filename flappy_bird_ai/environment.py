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


# ─── Asset paths (same layout as the reference project) ─────────────
_ASSET_BG        = "assets/sprites/background-day.png"
_ASSET_BASE      = "assets/sprites/base.png"
_ASSET_PIPE      = "assets/sprites/pipe-green.png"
_ASSET_BIRD_UP   = "assets/sprites/bluebird-upflap.png"
_ASSET_BIRD_MID  = "assets/sprites/bluebird-midflap.png"
_ASSET_BIRD_DOWN = "assets/sprites/bluebird-downflap.png"

_PIPE_SPRITE_H   = 500


class _SpriteCache:
    """Lazy-load and cache all pygame surfaces once."""
    _instance: "_SpriteCache | None" = None

    def __init__(self) -> None:
        self.bg:        pygame.Surface | None = None
        self.base:      pygame.Surface | None = None
        self.pipe:      pygame.Surface | None = None
        self.pipe_flip: pygame.Surface | None = None
        self.bird:      list[pygame.Surface]  = []
        self.font:      pygame.font.Font | None = None
        self._loaded    = False

    @classmethod
    def get(cls) -> "_SpriteCache":
        if cls._instance is None:
            cls._instance = _SpriteCache()
        return cls._instance

    def load(self, screen_w: int, screen_h: int,
             pipe_w: int, ground_h: int) -> None:
        if self._loaded:
            return
        self._loaded = True

        def safe(path: str) -> pygame.Surface | None:
            try:
                return pygame.image.load(path).convert_alpha()
            except Exception:
                return None

        raw = safe(_ASSET_BG)
        if raw:
            self.bg = pygame.transform.scale(raw, (screen_w, screen_h))

        raw = safe(_ASSET_BASE)
        if raw:
            # 2× wide so the scroll seam is always off-screen
            self.base = pygame.transform.scale(raw, (screen_w * 2, ground_h))

        raw = safe(_ASSET_PIPE)
        if raw:
            self.pipe      = pygame.transform.scale(raw, (pipe_w, _PIPE_SPRITE_H))
            self.pipe_flip = pygame.transform.flip(self.pipe, False, True)

        for path in (_ASSET_BIRD_UP, _ASSET_BIRD_MID, _ASSET_BIRD_DOWN):
            raw = safe(path)
            if raw:
                self.bird.append(raw)

        try:
            self.font = pygame.font.SysFont("impact", 40, bold=True)
        except Exception:
            self.font = pygame.font.SysFont(None, 40)


class FlappyBirdEnv:
    """A Flappy Bird game environment for reinforcement learning."""

    SCREEN_W    = 400
    SCREEN_H    = 600
    GRAVITY     = 0.5
    FLAP_STR    = -6
    PIPE_SPEED  = 3
    PIPE_GAP    = 180
    PIPE_WIDTH  = 60
    BIRD_X      = 50
    BIRD_RADIUS = 15
    GAP_MARGIN  = 20
    GROUND_H    = 100         # matches reference (GROUND_HEIGHT = 100)

    def __init__(self, seed: int | None = None) -> None:
        self.rng      = random.Random(seed) if seed is not None else random.Random()
        self.bird_y   = 0.0
        self.bird_vel = 0.0
        self.score    = 0
        self.pipes: List[Pipe] = []
        self._tick    = 0
        self._base_x  = 0        # ground scroll offset
        self._sprites = _SpriteCache.get()
        self._sprites_loaded = False

    # ─────────────────────────────────────────────────────────────────
    #  Core RL interface  (UNCHANGED from original)
    # ─────────────────────────────────────────────────────────────────

    def reset(self) -> np.ndarray:
        """Reset environment state for a new episode."""
        self.bird_y   = self.SCREEN_H / 2
        self.bird_vel = 0.0
        self.score    = 0
        self._tick    = 0
        self._base_x  = 0

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
        self._tick  += 1
        self._base_x = (self._base_x + self.PIPE_SPEED) % self.SCREEN_W

        for pipe in self.pipes:
            pipe["x"] -= self.PIPE_SPEED

        self._respawn_offscreen_pipes()

        pipe_passed = False
        for pipe in self.pipes:
            if (not pipe["passed"]) and (pipe["x"] + self.PIPE_WIDTH < self.BIRD_X):
                pipe["passed"] = True
                self.score    += 1
                pipe_passed    = True

        done = self._check_collision()

        if done:
            reward = REWARD_DEATH
        elif pipe_passed:
            reward = REWARD_PASS
        else:
            reward = REWARD_ALIVE

        # Proximity bonus
        if not done:
            nearest = self._nearest_pipe()
            dist_to_gap     = abs(self.bird_y - nearest["gap_y"]) / self.SCREEN_H
            proximity_bonus = 0.3 * max(0.0, 1.0 - dist_to_gap * 3)
            reward += proximity_bonus

        return self._get_state(), reward, done

    def _respawn_offscreen_pipes(self) -> None:
        if not self.pipes:
            return
        for pipe in self.pipes:
            if pipe["x"] + self.PIPE_WIDTH < 0:
                max_x          = max(p["x"] for p in self.pipes)
                pipe["x"]      = max_x + PIPE_SPACING
                pipe["gap_y"]  = self._random_gap_y()
                pipe["passed"] = False

    def _random_gap_y(self) -> int:
        return self.rng.randint(
            self.PIPE_GAP // 2 + self.GAP_MARGIN,
            self.SCREEN_H - self.GROUND_H - self.PIPE_GAP // 2 - self.GAP_MARGIN,
        )

    def _nearest_pipe(self) -> Pipe:
        ahead = [p for p in self.pipes
                 if p["x"] + self.PIPE_WIDTH >= self.BIRD_X - self.BIRD_RADIUS]
        if ahead:
            return min(ahead, key=lambda p: p["x"])
        return min(self.pipes, key=lambda p: p["x"])

    def _check_collision(self) -> bool:
        if self.bird_y - self.BIRD_RADIUS <= 0:
            return True
        if self.bird_y + self.BIRD_RADIUS >= self.SCREEN_H - self.GROUND_H:
            return True
        for pipe in self.pipes:
            px       = pipe["x"]
            gap_y    = pipe["gap_y"]
            pipe_top = gap_y - self.PIPE_GAP // 2
            pipe_bot = gap_y + self.PIPE_GAP // 2
            if (px < self.BIRD_X + self.BIRD_RADIUS and
                    px + self.PIPE_WIDTH > self.BIRD_X - self.BIRD_RADIUS):
                if self.bird_y - self.BIRD_RADIUS < pipe_top:
                    return True
                if self.bird_y + self.BIRD_RADIUS > pipe_bot:
                    return True
        return False

    def _get_state(self) -> np.ndarray:
        pipe     = self._nearest_pipe()
        pipe_x   = pipe["x"]
        gap_y    = pipe["gap_y"]
        pipe_top = gap_y - self.PIPE_GAP // 2
        pipe_bot = gap_y + self.PIPE_GAP // 2
        return np.array(
            [
                self.bird_y / self.SCREEN_H,
                (self.bird_vel + 10) / 20,
                (pipe_x - self.BIRD_X) / self.SCREEN_W,
                (self.bird_y - gap_y)  / self.SCREEN_H,
                (self.bird_y - pipe_top) / self.SCREEN_H,
                (pipe_bot - self.bird_y) / self.SCREEN_H,
            ],
            dtype=np.float32,
        )

    # ─────────────────────────────────────────────────────────────────
    #  Render  (sprite-based — uses the same assets/ folder)
    # ─────────────────────────────────────────────────────────────────

    def render(self, screen: pygame.Surface) -> None:
        """Draw the game using the original Flappy Bird sprite assets."""
        s = self._sprites
        if not self._sprites_loaded:
            s.load(self.SCREEN_W, self.SCREEN_H, self.PIPE_WIDTH, self.GROUND_H)
            self._sprites_loaded = True

        w, h       = self.SCREEN_W, self.SCREEN_H
        ground_y   = h - self.GROUND_H

        # ── 1. Background ─────────────────────────────────────────
        if s.bg:
            screen.blit(s.bg, (0, 0))
        else:
            screen.fill((113, 197, 207))

        # ── 2. Pipes ──────────────────────────────────────────────
        for pipe in self.pipes:
            px    = int(pipe["x"])
            gap_y = pipe["gap_y"]
            top_h = gap_y - self.PIPE_GAP // 2   # pixel where top pipe ends
            bot_y = gap_y + self.PIPE_GAP // 2   # pixel where bottom pipe starts

            if s.pipe and s.pipe_flip:
                # Top pipe: flipped sprite, bottom edge at top_h
                screen.blit(s.pipe_flip, (px, top_h - _PIPE_SPRITE_H))
                # Bottom pipe: normal sprite, top edge at bot_y
                screen.blit(s.pipe,      (px, bot_y))
            else:
                # Fallback solid rects
                pygame.draw.rect(screen, (115, 181, 54),
                                 (px, 0, self.PIPE_WIDTH, top_h))
                pygame.draw.rect(screen, (115, 181, 54),
                                 (px, bot_y, self.PIPE_WIDTH, h))

        # ── 3. Scrolling ground ───────────────────────────────────
        if s.base:
            screen.blit(s.base, (-self._base_x, ground_y))
        else:
            pygame.draw.rect(screen, (222, 216, 149), (0, ground_y, w, self.GROUND_H))

        # ── 4. Bird ───────────────────────────────────────────────
        # Cycle through up/mid/down frames (same 3-frame loop as reference)
        frame_idx = (self._tick // 5) % 3
        if s.bird:
            bird_surf = s.bird[min(frame_idx, len(s.bird) - 1)]
            tilt      = max(-30, min(35, int(-self.bird_vel * 3)))
            bird_rot  = pygame.transform.rotate(bird_surf, tilt)
            bird_rect = bird_rot.get_rect(center=(self.BIRD_X, int(self.bird_y)))
            screen.blit(bird_rot, bird_rect.topleft)
        else:
            pygame.draw.circle(screen, (255, 220, 50),
                               (self.BIRD_X, int(self.bird_y)), self.BIRD_RADIUS)

        # ── 5. Score (centered, white with black shadow) ──────────
        if s.font:
            txt    = str(self.score)
            shadow = s.font.render(txt, True, (0, 0, 0))
            main   = s.font.render(txt, True, (255, 255, 255))
            cx     = w // 2 - main.get_width() // 2
            screen.blit(shadow, (cx + 2, 22))
            screen.blit(main,   (cx,     20))