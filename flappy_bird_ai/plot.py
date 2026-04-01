"""
plot.py - Training progress visualization.

This module generates a matplotlib figure with two plots:
    1. Raw scores per episode (shows the noisy learning process)
    2. Rolling average scores (shows the smooth learning trend)

The resulting figure is saved as 'training_progress.png' and displayed.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_training(scores: list[int]) -> None:
    """
    Generate and display training progress graphs.

    Creates a side-by-side figure with:
        - Left: Raw score per episode (noisy, shows individual performance)
        - Right: Rolling average with window=50 (smooth trend line)

    Parameters:
        scores (list): List of integer scores (pipes passed) for each episode.
    """
    episodes = list(range(len(scores)))

    # ── Create figure with two subplots side by side ────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # ── Left plot: Raw scores per episode ───────────────────────────
    ax1.plot(episodes, scores, color='blue', alpha=0.4, linewidth=0.8)
    ax1.set_title("Score per Episode")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Pipes passed")
    ax1.grid(True, alpha=0.3)

    # ── Right plot: Rolling average (window=50) ─────────────────────
    # np.convolve with a uniform kernel computes the moving average
    window = 50
    if len(scores) >= window:
        # Create a uniform kernel of size 'window' that sums to 1
        kernel = np.ones(window) / window
        # 'valid' mode returns output only where the kernel fully overlaps
        rolling_avg = np.convolve(scores, kernel, mode='valid')
        # X-axis starts at 'window-1' because that's where the first full average is
        rolling_episodes = list(range(window - 1, len(scores)))
        ax2.plot(rolling_episodes, rolling_avg, color='red', linewidth=1.5)
    else:
        # Not enough data points for a rolling average — plot raw scores
        ax2.plot(episodes, scores, color='red', linewidth=1.5)

    ax2.set_title("Rolling Average (window=50)")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Pipes passed")
    ax2.grid(True, alpha=0.3)

    # ── Layout and save ─────────────────────────────────────────────
    plt.tight_layout()
    plt.savefig("training_progress.png", dpi=150)   # Save high-res image
    print("Training progress saved to 'training_progress.png'")
    plt.show()
