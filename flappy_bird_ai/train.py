"""train.py - Training loop for the DQN Flappy Bird agent."""

import os
import sys
from typing import List, Protocol, Tuple

import numpy as np
import pygame
import torch

from agent import DQNAgent
from config import (
    EPSILON_MIN,
    FPS,
    MAX_STEPS,
    NUM_EPISODES,
    PRINT_EVERY,
    RENDER_EVERY,
)
from environment import FlappyBirdEnv


class MetricsWriter(Protocol):
    def add_scalar(self, tag: str, scalar_value: float, global_step: int) -> None:
        ...

    def close(self) -> None:
        ...


class NoOpWriter:
    def add_scalar(self, tag: str, scalar_value: float, global_step: int) -> None:
        del tag, scalar_value, global_step

    def close(self) -> None:
        return


def create_metrics_writer(log_dir: str) -> MetricsWriter:
    """Create a TensorBoard writer when available, else a no-op writer."""
    try:
        from torch.utils.tensorboard import SummaryWriter

        return SummaryWriter(log_dir)
    except ModuleNotFoundError:
        print("TensorBoard not installed; continuing without metrics logging.")
        return NoOpWriter()


def train_agent(
    render: bool = False,
    episodes_override: int | None = None,
    resume: bool = False,
) -> Tuple[List[int], DQNAgent]:
    """Train the DQN agent to play Flappy Bird."""
    env = FlappyBirdEnv()
    agent = DQNAgent()

    if resume:
        model_path = "model.pth"
        if not os.path.exists(model_path):
            raise FileNotFoundError("Cannot resume: 'model.pth' not found.")
        agent.model.load_state_dict(
            torch.load(model_path, map_location=agent.device, weights_only=True)
        )
        agent.target_model.load_state_dict(agent.model.state_dict())
        agent.epsilon = EPSILON_MIN
        print("Resumed training from 'model.pth'")

    screen = None
    clock = None
    font = None
    if render:
        pygame.init()
        screen = pygame.display.set_mode((env.SCREEN_W, env.SCREEN_H))
        pygame.display.set_caption("Flappy Bird AI - Training")
        clock = pygame.time.Clock()
        font = pygame.font.SysFont(None, 24)

    total_episodes = episodes_override if episodes_override is not None else NUM_EPISODES
    scores: List[int] = []
    best_score = 0
    writer = create_metrics_writer("runs/flappy_bird")

    print("=" * 60)
    print("  FLAPPY BIRD DQN TRAINING")
    print("=" * 60)
    print(f"  Episodes: {total_episodes}")
    print(f"  Rendering: {'Every ' + str(RENDER_EVERY) + ' episodes' if render else 'Disabled'}")
    print(f"  Resume: {'Enabled' if resume else 'Disabled'}")
    print("=" * 60)

    for episode in range(total_episodes):
        state = env.reset()
        done = False
        step = 0

        should_render = render and (episode % RENDER_EVERY == 0)

        while not done and step < MAX_STEPS:
            if should_render:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        writer.close()
                        pygame.quit()
                        sys.exit()

            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)

            agent.memory.push(state, action, reward, next_state, float(done))
            agent.train()

            state = next_state
            step += 1

            if should_render:
                if screen is None or clock is None or font is None:
                    raise RuntimeError("Rendering resources were not initialized.")
                screen.fill((0, 0, 0))
                env.render(screen)

                ep_text = font.render(f"Episode: {episode}", True, (255, 255, 255))
                screen.blit(ep_text, (10, 35))

                eps_text = font.render(
                    f"Epsilon: {agent.epsilon:.3f}", True, (180, 180, 180)
                )
                screen.blit(eps_text, (10, 60))

                pygame.display.flip()
                clock.tick(FPS)

        agent.decay_epsilon()
        scores.append(env.score)

        last_100 = scores[-100:] if len(scores) >= 100 else scores
        avg = float(np.mean(last_100))

        writer.add_scalar("Score", float(env.score), episode)
        writer.add_scalar("Epsilon", float(agent.epsilon), episode)
        writer.add_scalar("Average Score (100)", avg, episode)

        if env.score > best_score:
            best_score = env.score
            torch.save(agent.model.state_dict(), "best_model.pth")
            print(f"  New best score: {best_score} - model saved to 'best_model.pth'")

        if episode > 0 and episode % 500 == 0:
            checkpoint_name = f"checkpoint_ep{episode}.pth"
            torch.save(agent.model.state_dict(), checkpoint_name)
            print(f"  Safety checkpoint saved: '{checkpoint_name}'")

        if episode % PRINT_EVERY == 0:
            print(
                f"  Episode {episode:>5d} | "
                f"Score: {env.score:>4d} | "
                f"Avg(100): {avg:>6.1f} | "
                f"Epsilon: {agent.epsilon:.3f}"
            )

    writer.close()

    print("=" * 60)
    print("  TRAINING COMPLETE")
    print("=" * 60)

    if render and pygame.get_init():
        pygame.quit()

    return scores, agent
