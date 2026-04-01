"""
train.py - Training loop for the DQN Flappy Bird agent.

This module orchestrates the training process:
    1. Creates the environment and agent
    2. Runs episodes where the agent interacts with the game
    3. Stores experiences and trains the network each step
    4. Optionally renders the game every N episodes
    5. Prints progress statistics periodically

The training loop follows the standard RL cycle:
    observe state -> choose action -> receive reward -> learn -> repeat
"""

import sys
import numpy as np
import pygame

from environment import FlappyBirdEnv
from agent import DQNAgent
from config import (
    NUM_EPISODES, MAX_STEPS, FPS,
    RENDER_EVERY, PRINT_EVERY
)


def train_agent(render=False):
    """
    Train the DQN agent to play Flappy Bird.

    Parameters:
        render (bool): If True, show the game window every RENDER_EVERY episodes.
                       If False, train silently (much faster).

    Returns:
        tuple: (scores, agent)
            - scores (list): List of scores (pipes passed) for each episode.
            - agent (DQNAgent): The trained agent (for saving the model).
    """
    # ── Initialize environment and agent ────────────────────────────
    env = FlappyBirdEnv()
    agent = DQNAgent()

    # ── Set up pygame display if rendering is enabled ───────────────
    screen = None
    clock = None
    if render:
        pygame.init()
        screen = pygame.display.set_mode((env.SCREEN_W, env.SCREEN_H))
        pygame.display.set_caption("Flappy Bird AI - Training")
        clock = pygame.time.Clock()

    scores = []     # Track score (pipes passed) for each episode

    print("=" * 60)
    print("  FLAPPY BIRD DQN TRAINING")
    print("=" * 60)
    print(f"  Episodes: {NUM_EPISODES}")
    print(f"  Rendering: {'Every ' + str(RENDER_EVERY) + ' episodes' if render else 'Disabled'}")
    print("=" * 60)

    # ── Main training loop: one iteration = one episode ─────────────
    for episode in range(NUM_EPISODES):
        state = env.reset()         # Reset game for new episode
        done = False
        total_reward = 0.0
        step = 0

        # Decide whether to render this episode
        should_render = render and (episode % RENDER_EVERY == 0)

        # ── Episode loop: one iteration = one game frame ────────────
        while not done and step < MAX_STEPS:

            # ── Handle pygame events (close window, etc.) ───────────
            if should_render:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()

            # ── Agent chooses an action based on current state ──────
            action = agent.choose_action(state)

            # ── Environment executes the action and returns results ──
            next_state, reward, done = env.step(action)

            # ── Store the experience in replay memory ───────────────
            # done is stored as float (1.0 or 0.0) for tensor math
            agent.memory.push(state, action, reward, next_state, float(done))

            # ── Train the network on a batch from memory ────────────
            agent.train()

            # ── Move to next state ──────────────────────────────────
            state = next_state
            total_reward += reward
            step += 1

            # ── Render the game if this is a render episode ─────────
            if should_render:
                screen.fill((0, 0, 0))          # Clear screen
                env.render(screen)               # Draw game state

                # Draw episode info overlay
                font = pygame.font.SysFont(None, 24)

                # Episode number
                ep_text = font.render(f"Episode: {episode}", True, (255, 255, 255))
                screen.blit(ep_text, (10, 35))

                # Current epsilon value (exploration rate)
                eps_text = font.render(
                    f"Epsilon: {agent.epsilon:.3f}", True, (180, 180, 180)
                )
                screen.blit(eps_text, (10, 60))

                pygame.display.flip()            # Update the display
                clock.tick(FPS)                   # Cap at target FPS

        # ── End of episode: decay exploration rate ──────────────────
        agent.decay_epsilon()
        scores.append(env.score)

        # ── Print progress statistics every PRINT_EVERY episodes ────
        if episode % PRINT_EVERY == 0:
            # Calculate average score over last 100 episodes
            last_100 = scores[-100:] if len(scores) >= 100 else scores
            avg = np.mean(last_100)

            print(
                f"  Episode {episode:>5d} | "
                f"Score: {env.score:>4d} | "
                f"Avg(100): {avg:>6.1f} | "
                f"Epsilon: {agent.epsilon:.3f}"
            )

    print("=" * 60)
    print("  TRAINING COMPLETE")
    print("=" * 60)

    # Clean up pygame if it was initialized for rendering
    if render and pygame.get_init():
        pygame.quit()

    return scores, agent
