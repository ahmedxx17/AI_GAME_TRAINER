"""
main.py - Entry point for the Flappy Bird AI project.

Usage:
    python main.py --train            Train the agent (no game window)
    python main.py --train --render   Train with the game window visible
    python main.py --play             Load a trained model and watch the AI play

This file handles command-line argument parsing and dispatches to the
appropriate mode (training or playing).
"""

import sys
import argparse
import numpy as np
import torch
import pygame

from environment import FlappyBirdEnv
from agent import DQNAgent
from train import train_agent
from plot import plot_training
from config import FPS


def play_mode():
    """
    Load a trained model and let the AI play Flappy Bird in real-time.

    The game runs with rendering enabled and epsilon set to 0
    (pure exploitation — the agent always picks the best action
    according to its learned Q-values, with no random exploration).

    The game loops until the user closes the window.
    """
    # ── Initialize agent and load trained weights ───────────────────
    agent = DQNAgent()
    try:
        agent.model.load_state_dict(
            torch.load("model.pth", map_location=agent.device, weights_only=True)
        )
        print("Loaded trained model from 'model.pth'")
    except FileNotFoundError:
        print("ERROR: 'model.pth' not found!")
        print("Train the agent first: python main.py --train")
        sys.exit(1)

    # Set epsilon to 0: no exploration, pure exploitation of learned knowledge
    agent.epsilon = 0.0
    agent.model.eval()      # Set model to evaluation mode (disables dropout etc.)

    # ── Initialize pygame ───────────────────────────────────────────
    pygame.init()
    env = FlappyBirdEnv()
    screen = pygame.display.set_mode((env.SCREEN_W, env.SCREEN_H))
    pygame.display.set_caption("Flappy Bird AI - Watch Me Play!")
    clock = pygame.time.Clock()

    print("=" * 60)
    print("  WATCHING TRAINED AI PLAY")
    print("  Close the window to exit.")
    print("=" * 60)

    # ── Main game loop ──────────────────────────────────────────────
    running = True
    while running:
        state = env.reset()
        done = False

        while not done and running:
            # ── Handle events ───────────────────────────────────────
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break

            if not running:
                break

            # ── Agent chooses action (no randomness) ────────────────
            action = agent.choose_action(state)
            state, reward, done = env.step(action)

            # ── Render the game ─────────────────────────────────────
            screen.fill((0, 0, 0))
            env.render(screen)

            # Draw "AI PLAYING" indicator
            font = pygame.font.SysFont(None, 24)
            ai_text = font.render("AI Playing", True, (100, 255, 100))
            screen.blit(ai_text, (10, 35))

            pygame.display.flip()
            clock.tick(FPS)

        if running:
            print(f"  Game Over! Score: {env.score}")

    pygame.quit()
    print("Bye!")


def main():
    """
    Parse command-line arguments and run the appropriate mode.
    """
    parser = argparse.ArgumentParser(
        description="Flappy Bird AI - Deep Q-Learning Agent"
    )
    parser.add_argument(
        "--train", action="store_true",
        help="Train the agent from scratch"
    )
    parser.add_argument(
        "--render", action="store_true",
        help="Show the game window during training (slower but visual)"
    )
    parser.add_argument(
        "--play", action="store_true",
        help="Load a trained model and watch the AI play"
    )

    args = parser.parse_args()

    # ── Validate arguments ──────────────────────────────────────────
    if not args.train and not args.play:
        parser.print_help()
        print("\nExample usage:")
        print("  python main.py --train            # Train the AI")
        print("  python main.py --train --render    # Train with visuals")
        print("  python main.py --play              # Watch trained AI play")
        sys.exit(0)

    # ── Training mode ───────────────────────────────────────────────
    if args.train:
        print("\nStarting training...\n")
        scores, agent = train_agent(render=args.render)

        # Save the trained model weights
        torch.save(agent.model.state_dict(), "model.pth")
        print("\nModel saved to 'model.pth'")

        # ── Print final statistics ──────────────────────────────────
        print("\n" + "=" * 60)
        print("  FINAL TRAINING STATISTICS")
        print("=" * 60)
        print(f"  Best score:        {max(scores)}")
        print(f"  Average score:     {np.mean(scores):.1f}")
        last_100 = scores[-100:] if len(scores) >= 100 else scores
        print(f"  Last-100 average:  {np.mean(last_100):.1f}")
        print("=" * 60)

        # Generate training progress plots
        plot_training(scores)

    # ── Play mode ───────────────────────────────────────────────────
    elif args.play:
        play_mode()


if __name__ == "__main__":
    main()
