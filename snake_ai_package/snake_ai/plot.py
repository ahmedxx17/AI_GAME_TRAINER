"""plot.py - Training progress visualisation for the Snake CNN-DQN agent."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_training(scores: list[int]) -> None:
    """Save a two-panel training-progress figure to 'training_progress.png'."""
    episodes = list(range(len(scores)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # ── Raw scores ──────────────────────────────────────────────────
    ax1.plot(episodes, scores, color="blue", alpha=0.4, linewidth=0.8)
    ax1.set_title("Score per Episode")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Food eaten")
    ax1.grid(True, alpha=0.3)

    # ── Rolling average ─────────────────────────────────────────────
    window = 50
    if len(scores) >= window:
        kernel = np.ones(window) / window
        rolling = np.convolve(scores, kernel, mode="valid")
        ax2.plot(range(window - 1, len(scores)), rolling, color="red", linewidth=1.5)
    else:
        ax2.plot(episodes, scores, color="red", linewidth=1.5)

    ax2.set_title("Rolling Average (window=50)")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Food eaten")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("training_progress.png", dpi=150)
    print("Training progress saved to 'training_progress.png'")
