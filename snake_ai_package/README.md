# AI Game Trainer

A collection of AI agents that learn to play games using Deep Reinforcement Learning.  
Two different approaches are demonstrated: **feature-based DQN** and **vision-based CNN-DQN**.

## Games

### Flappy Bird — Feature-Based DQN
A DQN agent that learns to play Flappy Bird using **6 hand-crafted state features** (bird position, velocity, pipe distances) fed into a 2-layer MLP.  
See [flappy_bird_ai/README.md](flappy_bird_ai/README.md) for full details.

### Snake — Vision-Based CNN-DQN
A CNN-DQN agent that learns to play Snake from **raw 84×84 pixel frames** — no hand-crafted features. Four consecutive frames are stacked as input channels so the CNN can perceive motion. This follows the approach from the [DeepMind Atari DQN paper](https://www.nature.com/articles/nature14236).  
See [snake_ai/README.md](snake_ai/README.md) for full details.

## Approach Comparison

| | Flappy Bird | Snake |
|---|---|---|
| **Input** | 6 hand-crafted features | Raw 84×84 pixel frames |
| **Network** | 2-layer MLP (128 hidden) | 3-layer CNN + 2-layer FC |
| **Parameters** | ~18K | ~424K |
| **State representation** | Engineered distances & velocity | Raw pixel intensities |
| **Learning approach** | Feature engineering + RL | End-to-end visual learning |

## Quick Start

```bash
pip install -r requirements.txt

# Flappy Bird (feature-based)
cd flappy_bird_ai
python main.py --train
python main.py --play

# Snake (vision-based)
cd snake_ai
python main.py --train
python main.py --play
```
