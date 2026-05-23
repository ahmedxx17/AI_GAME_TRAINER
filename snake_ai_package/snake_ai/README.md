# Snake AI — Vision-Based CNN-DQN

A Deep Q-Network agent that learns to play Snake using **raw pixel input** through a Convolutional Neural Network — the AI "sees" the game and learns to play from visual information alone.

## How It Works

Unlike the Flappy Bird agent (which receives 6 hand-crafted numerical features), this Snake agent receives **raw 84×84 grayscale frames** of the game. Four consecutive frames are stacked together so the CNN can perceive motion (e.g. which direction the snake is heading).

### Input Pipeline

```
Game State  →  Render as pixels  →  Grayscale 84×84  →  Stack 4 frames  →  CNN
```

### CNN Architecture (DeepMind Atari-style)

```
Input: 4 × 84 × 84 (4 stacked grayscale frames)
  ↓
Conv2d(4 → 16, kernel 8×8, stride 4)  → ReLU     — coarse spatial features
  ↓
Conv2d(16 → 32, kernel 4×4, stride 2) → ReLU     — mid-level features
  ↓
Conv2d(32 → 32, kernel 3×3, stride 1) → ReLU     — fine-grained features
  ↓
Flatten (32 × 7 × 7 = 1,568)
  ↓
Linear(1568 → 256) → ReLU
  ↓
Linear(256 → 4)  — Q-values for [Up, Right, Down, Left]
```

**Total parameters: ~424K** (vs ~18K for the Flappy Bird MLP — 23× larger)

## Training Results

Trained for **5,000 episodes** on CPU:

| Metric | Value |
|---|---|
| **Best score** | 28 food items (on a 10×10 grid) |
| **Last-100 average** | 11.6 |
| **Overall average** | 7.4 |

The agent learns to navigate the grid, seek food efficiently, and avoid collisions with walls and its own body — all from raw pixel input alone.

## Techniques Used

| Technique | Purpose |
|---|---|
| **Frame Stacking (×4)** | Gives the CNN temporal information (direction of movement) |
| **Target Network** | Stabilises training by decoupling action selection from target computation |
| **Huber Loss** | More robust to outlier rewards than MSE |
| **Gradient Clipping** | Prevents destructively large parameter updates |
| **Reward Shaping** | Manhattan-distance signal guides exploration toward food |
| **Hunger Timeout** | Prevents the snake from surviving indefinitely without eating |
| **Epsilon-Greedy Decay** | Balances exploration (random) vs exploitation (learned) |

## Usage

```bash
cd snake_ai
python main.py --train              # Train the CNN-DQN
python main.py --train --render     # Train with visual output
python main.py --train --resume     # Continue from existing model
python main.py --play               # Watch the trained AI play
python main.py --episodes 5000      # Override episode count
```

## Comparison: Flappy Bird vs Snake

| | Flappy Bird | Snake |
|---|---|---|
| **Input** | 6 hand-crafted features | Raw 84×84 pixel frames |
| **Network** | 2-layer MLP (128 hidden) | 3-layer CNN + 2-layer FC |
| **Parameters** | ~18K | ~424K |
| **State info** | Distances, velocity | Visual: pixel intensities |
| **Approach** | Feature engineering | End-to-end visual learning |
