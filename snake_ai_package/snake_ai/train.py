"""train.py - Training loop for the Snake CNN-DQN agent."""

import os
import sys
from typing import List, Protocol, Tuple

import numpy as np
import torch

from agent import CNNAgent
from config import (
    DISPLAY_SIZE,
    EPSILON_MIN,
    FPS,
    MAX_STEPS,
    NUM_EPISODES,
    PRINT_EVERY,
    RENDER_EVERY,
    TRAIN_EVERY,
    TRAIN_START,
)
from environment import SnakeEnv


# ─── TensorBoard helpers ───────────────────────────────────────────

class MetricsWriter(Protocol):
    def add_scalar(self, tag: str, scalar_value: float, global_step: int) -> None: ...
    def close(self) -> None: ...


class NoOpWriter:
    """Fallback when TensorBoard is not installed."""
    def add_scalar(self, tag: str, scalar_value: float, global_step: int) -> None:
        del tag, scalar_value, global_step

    def close(self) -> None:
        return


def _create_writer(log_dir: str) -> MetricsWriter:
    try:
        from torch.utils.tensorboard import SummaryWriter
        return SummaryWriter(log_dir)
    except ModuleNotFoundError:
        print("TensorBoard not installed; continuing without metrics logging.")
        return NoOpWriter()


# ─── Main training function ────────────────────────────────────────

def train_agent(
    render: bool = False,
    episodes_override: int | None = None,
    resume: bool = False,
) -> Tuple[List[int], CNNAgent]:
    """Train the CNN-DQN agent to play Snake from pixel input."""

    env   = SnakeEnv()
    agent = CNNAgent()

    if resume:
        path = "model.pth"
        if not os.path.exists(path):
            raise FileNotFoundError("Cannot resume: 'model.pth' not found.")
        agent.model.load_state_dict(
            torch.load(path, map_location=agent.device, weights_only=True)
        )
        agent.target_model.load_state_dict(agent.model.state_dict())
        agent.epsilon = EPSILON_MIN
        print("Resumed training from 'model.pth'")

    # ── Pygame setup (only when rendering) ──────────────────────────
    screen = None
    clock  = None
    font   = None
    if render:
        import pygame
        pygame.init()
        screen = pygame.display.set_mode((DISPLAY_SIZE, DISPLAY_SIZE))
        pygame.display.set_caption("Snake AI — CNN-DQN Training")
        clock = pygame.time.Clock()
        font  = pygame.font.SysFont(None, 24)

    total_episodes = episodes_override if episodes_override is not None else NUM_EPISODES
    scores: List[int] = []
    best_score  = 0
    total_steps = 0
    writer = _create_writer("runs/snake")

    print("=" * 60)
    print("  SNAKE CNN-DQN TRAINING")
    print("=" * 60)
    print(f"  Episodes:  {total_episodes}")
    print(f"  Rendering: {'Every ' + str(RENDER_EVERY) + ' episodes' if render else 'Disabled'}")
    print(f"  Resume:    {'Enabled' if resume else 'Disabled'}")
    print("=" * 60)

    for episode in range(total_episodes):
        frame = env.reset()
        agent.reset_frames(frame)
        state = agent.get_state()
        done  = False
        step  = 0

        should_render = render and (episode % RENDER_EVERY == 0)

        while not done and step < MAX_STEPS:
            # ── Handle window events ────────────────────────────────
            if should_render:
                import pygame
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        writer.close()
                        pygame.quit()
                        sys.exit()

            # ── Agent acts ──────────────────────────────────────────
            action = agent.choose_action(state)
            next_frame, reward, done = env.step(action)
            agent.push_frame(next_frame)
            next_state = agent.get_state()

            agent.memory.push(state, action, reward, next_state, float(done))

            # ── Train every N steps once memory is warm ─────────────
            if total_steps >= TRAIN_START and total_steps % TRAIN_EVERY == 0:
                agent.train()

            state = next_state
            step  += 1
            total_steps += 1

            # ── Render current frame ────────────────────────────────
            if should_render:
                import pygame
                if screen is None or clock is None or font is None:
                    raise RuntimeError("Rendering resources not initialised.")
                screen.fill((0, 0, 0))
                env.render(screen)
                ep_text  = font.render(f"Episode: {episode}", True, (255, 255, 255))
                eps_text = font.render(f"Epsilon: {agent.epsilon:.3f}", True, (180, 180, 180))
                screen.blit(ep_text,  (10, DISPLAY_SIZE - 50))
                screen.blit(eps_text, (10, DISPLAY_SIZE - 30))
                pygame.display.flip()
                clock.tick(FPS)

        # ── End of episode bookkeeping ──────────────────────────────
        agent.decay_epsilon()
        scores.append(env.score)

        last_100 = scores[-100:] if len(scores) >= 100 else scores
        avg = float(np.mean(last_100))

        writer.add_scalar("Score",              float(env.score),    episode)
        writer.add_scalar("Epsilon",            float(agent.epsilon), episode)
        writer.add_scalar("Average Score (100)", avg,                 episode)

        if env.score > best_score:
            best_score = env.score
            torch.save(agent.model.state_dict(), "best_model.pth")
            print(f"  ** New best score: {best_score} — saved best_model.pth")

        if episode > 0 and episode % 500 == 0:
            ckpt = f"checkpoint_ep{episode}.pth"
            torch.save(agent.model.state_dict(), ckpt)
            print(f"  Checkpoint saved: '{ckpt}'")

        if episode % PRINT_EVERY == 0:
            print(
                f"  Episode {episode:>5d} | "
                f"Score: {env.score:>3d} | "
                f"Avg(100): {avg:>5.1f} | "
                f"Eps: {agent.epsilon:.3f} | "
                f"Steps: {step:>4d} | "
                f"Mem: {len(agent.memory):>6d}"
            )

    writer.close()
    if render:
        import pygame
        pygame.quit()

    return scores, agent
