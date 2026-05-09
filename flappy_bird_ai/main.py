"""main.py - Entry point for the Flappy Bird AI project."""

import argparse
import os
import sys

import numpy as np
import pygame
import torch

from agent import DQNAgent
from config import FPS
from environment import FlappyBirdEnv
from plot import plot_training
from train import train_agent


def play_mode() -> None:
    """Load a trained model and let the AI play Flappy Bird in real-time."""
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

    agent.target_model.load_state_dict(agent.model.state_dict())
    agent.epsilon = 0.0
    agent.model.eval()

    pygame.init()
    env = FlappyBirdEnv()
    screen = pygame.display.set_mode((env.SCREEN_W, env.SCREEN_H))
    pygame.display.set_caption("Flappy Bird AI - Watch Me Play!")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)

    print("=" * 60)
    print("  WATCHING TRAINED AI PLAY")
    print("  Close the window to exit.")
    print("=" * 60)

    running = True
    while running:
        state = env.reset()
        done = False

        while not done and running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break

            if not running:
                break

            action = agent.choose_action(state)
            state, _, done = env.step(action)

            screen.fill((0, 0, 0))
            env.render(screen)

            ai_text = font.render("AI Playing", True, (100, 255, 100))
            screen.blit(ai_text, (10, 35))

            pygame.display.flip()
            clock.tick(FPS)

        if running:
            print(f"  Game Over! Score: {env.score}")

    pygame.quit()
    print("Bye!")


def main() -> None:
    """Parse command-line arguments and run training or play mode."""
    parser = argparse.ArgumentParser(
        description="Flappy Bird AI - Deep Q-Learning Agent"
    )
    parser.add_argument(
        "--train", action="store_true", help="Train the agent from scratch"
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Show the game window during training (slower but visual)",
    )
    parser.add_argument(
        "--play", action="store_true", help="Load a trained model and watch the AI play"
    )
    parser.add_argument(
        "--vs", action="store_true", help="Race against the fully trained AI in split-screen mode"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from an existing model.pth",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Override NUM_EPISODES from config.py",
    )

    args = parser.parse_args()

    if not args.train and not args.play and not args.vs:
        parser.print_help()
        print("\nExample usage:")
        print("  python main.py --train             # Train the AI")
        print("  python main.py --train --render    # Train with visuals")
        print("  python main.py --train --resume    # Continue from model.pth")
        print("  python main.py --play              # Watch trained AI play")
        print("  python main.py --vs                # Race against the AI")
        sys.exit(0)

    if args.vs:
        import human_vs_ai
        human_vs_ai.run_vs_mode()
        sys.exit(0)

    if args.train:
        if args.episodes is not None and args.episodes <= 0:
            print("ERROR: --episodes must be greater than 0")
            sys.exit(1)

        if args.resume and not os.path.exists("model.pth"):
            print("ERROR: --resume requested but 'model.pth' does not exist.")
            print("Run without --resume first, or provide a model file.")
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
