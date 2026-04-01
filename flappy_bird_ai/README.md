# Flappy Bird AI - Deep Q-Learning Agent

An AI agent that learns to play Flappy Bird from scratch using Deep Q-Learning (DQN). The agent starts with zero knowledge of the game — no hardcoded rules, no human guidance — and learns entirely through trial and error. After training, it plays the game indefinitely without dying.

This is a beginner-to-intermediate level Reinforcement Learning project built with PyTorch, pygame, and numpy.

---

## How It Works

### Reinforcement Learning (RL)

The AI agent interacts with the game environment in a loop:

1. **Observe** the current state (bird position, velocity, pipe distance, etc.)
2. **Choose an action** (flap or do nothing)
3. **Receive a reward** (+1 for passing a pipe, +0.1 for surviving, -1 for crashing)
4. **Learn** from the experience to make better decisions next time

### Deep Q-Network (DQN)

A neural network estimates the **Q-value** (expected future reward) for each action in a given state. The agent picks the action with the highest Q-value. The network is trained using:

- **Experience Replay**: Stores past experiences and trains on random batches, which breaks temporal correlations and stabilizes learning.
- **Epsilon-Greedy Exploration**: Starts with random actions (exploration) and gradually shifts to using learned knowledge (exploitation).
- **Bellman Equation**: Updates Q-values by combining the immediate reward with the discounted future value.

---

## Installation

### Prerequisites

- Python 3.10 or higher
- pip (Python package installer)

### Install Dependencies

```bash
pip install pygame torch numpy matplotlib
```

> **Note**: If you have an NVIDIA GPU, install the CUDA version of PyTorch for faster training. See [pytorch.org](https://pytorch.org/) for instructions.

---

## How to Run

### Train the AI (headless - fastest)

```bash
python main.py --train
```

### Train with game window visible

```bash
python main.py --train --render
```

The game window appears every 50 episodes so you can watch the AI's progress.

### Watch the trained AI play

```bash
python main.py --play
```

This loads the saved `model.pth` file and lets the AI play with no randomness.

---

## What to Expect

### Training Progress

| Episode Range | Expected Behavior |
|---------------|-------------------|
| 1 - 50 | Score = 0, dies immediately (random actions) |
| 100 - 200 | Starts reaching scores of 1-3 |
| 500+ | Consistent scores of 5-15 |
| 1000+ | Scores of 20-50+ |
| 2000 | Should play indefinitely or achieve very high scores |

Training 2000 episodes takes approximately:
- **Without rendering**: 5-15 minutes (depending on hardware)
- **With rendering**: 30-60+ minutes (rendering slows things down)

### Training Output

A `training_progress.png` file is saved at the end of training showing:
- Raw scores per episode (noisy learning curve)
- Rolling average scores (smooth trend showing improvement)

---

## File Structure

```
flappy_bird_ai/
|
|-- main.py           Entry point. Parse args, dispatch to train or play mode.
|-- environment.py    FlappyBirdEnv class. Game physics, collision, rendering.
|-- agent.py          DQNAgent class. Action selection, training, exploration.
|-- model.py          DQN neural network. 4 -> 64 -> 64 -> 2 architecture.
|-- memory.py         ReplayMemory class. Fixed-size experience buffer.
|-- config.py         All hyperparameters in one place for easy tuning.
|-- train.py          Training loop. Runs episodes, collects data, trains.
|-- plot.py           Matplotlib visualization of training progress.
|-- README.md         This file.
```

### How the files connect:

```
main.py
  |-- train.py (imports train_agent function)
  |     |-- environment.py (creates game environment)
  |     |-- agent.py (creates AI agent)
  |           |-- model.py (neural network used by agent)
  |           |-- memory.py (replay buffer used by agent)
  |           |-- config.py (hyperparameters used everywhere)
  |-- plot.py (generates training graphs)
```

---

## Key Hyperparameters

All hyperparameters are defined in `config.py`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `NUM_EPISODES` | 2000 | Total training episodes |
| `BATCH_SIZE` | 32 | Experiences per training step |
| `LEARNING_RATE` | 0.001 | How fast the network learns |
| `GAMMA` | 0.99 | How much the agent values future rewards |
| `EPSILON_START` | 1.0 | Initial exploration rate (100% random) |
| `EPSILON_MIN` | 0.01 | Minimum exploration rate (1% random) |
| `EPSILON_DECAY` | 0.995 | How fast exploration decreases |
| `MEMORY_CAPACITY` | 10000 | Max experiences in replay buffer |

### Tuning Tips

- **Not learning?** Try increasing `MEMORY_CAPACITY` or decreasing `EPSILON_DECAY` (slower decay).
- **Learning too slow?** Try increasing `LEARNING_RATE` slightly (e.g., 0.002).
- **Unstable?** Try decreasing `LEARNING_RATE` or increasing `BATCH_SIZE`.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: No module named 'pygame'` | Run `pip install pygame` |
| `ModuleNotFoundError: No module named 'torch'` | Run `pip install torch` |
| `model.pth not found` when using `--play` | Train first with `python main.py --train` |
| AI not improving after 300 episodes | Check state normalization, reward values, and epsilon decay |
| Game window not responding | Make sure you're running with `--render` flag |

---

## License

This project is for educational purposes. Feel free to use, modify, and share.
