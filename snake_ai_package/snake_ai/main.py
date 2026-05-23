"""main.py - Entry point for the Snake AI project."""

import argparse
import os
import sys

import numpy as np
import torch

from agent import CNNAgent
from config import DISPLAY_SIZE, FPS
from environment import SnakeEnv
from plot import plot_training
from train import train_agent


def play_mode() -> None:
    """Load a trained model and let the CNN-DQN play Snake in real-time."""
    agent = CNNAgent()
    try:
        agent.model.load_state_dict(
            torch.load("model.pth", map_location=agent.device, weights_only=True)
        )
        print("Loaded trained model from 'model.pth'")
    except FileNotFoundError:
        print("ERROR: 'model.pth' not found!")
        print("Train the agent first: python main.py --train")
        sys.exit(1)

    agent.target_model.load_state_dict(agent.model.state_dict())
    agent.epsilon = 0.0
    agent.model.eval()

    import pygame
    pygame.init()
    env    = SnakeEnv()
    screen = pygame.display.set_mode((DISPLAY_SIZE, DISPLAY_SIZE))
    pygame.display.set_caption("Snake AI — CNN-DQN Playing")
    clock  = pygame.time.Clock()

    print("=" * 60)
    print("  WATCHING TRAINED CNN-DQN PLAY SNAKE")
    print("  Close the window to exit.")
    print("=" * 60)

    running = True
    while running:
        frame = env.reset()
        agent.reset_frames(frame)
        done = False

        while not done and running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break
            if not running:
                break

            state  = agent.get_state()
            action = agent.choose_action(state)
            frame, _, done = env.step(action)
            agent.push_frame(frame)

            screen.fill((0, 0, 0))
            env.render(screen)
            pygame.display.flip()
            clock.tick(FPS)

        if running:
            print(f"  Game Over! Score: {env.score}")

    pygame.quit()
    print("Bye!")


def main() -> None:
    """Parse CLI arguments and dispatch to training or play mode."""
    parser = argparse.ArgumentParser(
        description="Snake AI — Vision-Based CNN-DQN Agent"
    )
    parser.add_argument(
        "--train", action="store_true", help="Train the CNN-DQN agent from scratch"
    )
    parser.add_argument(
        "--render", action="store_true",
        help="Show the game window during training (slower but visual)",
    )
    parser.add_argument(
        "--play", action="store_true",
        help="Load a trained model and watch the AI play",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume training from an existing model.pth",
    )
    parser.add_argument(
        "--episodes", type=int, default=None,
        help="Override NUM_EPISODES from config.py",
    )

    args = parser.parse_args()

    if not args.train and not args.play:
        parser.print_help()
        print("\nExample usage:")
        print("  python main.py --train              # Train the CNN-DQN")
        print("  python main.py --train --render     # Train with visuals")
        print("  python main.py --train --resume     # Continue from model.pth")
        print("  python main.py --play               # Watch trained AI play")
        sys.exit(0)

    if args.train:
        if args.episodes is not None and args.episodes <= 0:
            print("ERROR: --episodes must be greater than 0")
            sys.exit(1)
        if args.resume and not os.path.exists("model.pth"):
            print("ERROR: --resume requested but 'model.pth' does not exist.")
            sys.exit(1)

        print("\nStarting training...\n")
        scores, agent = train_agent(
            render=args.render,
            episodes_override=args.episodes,
            resume=args.resume,
        )

        torch.save(agent.model.state_dict(), "model.pth")
        print("\nModel saved to 'model.pth'")

        print("\n" + "=" * 60)
        print("  FINAL TRAINING STATISTICS")
        print("=" * 60)
        print(f"  Best score:        {max(scores)}")
        print(f"  Average score:     {np.mean(scores):.1f}")
        last_100 = scores[-100:] if len(scores) >= 100 else scores
        print(f"  Last-100 average:  {np.mean(last_100):.1f}")
        print("=" * 60)

        plot_training(scores)

    elif args.play:
        play_mode()


if __name__ == "__main__":
    main()
