# 🎮 AI Game Trainer: Flappy Bird AI

A complete Deep Q-Learning implementation that trains an AI agent to master the Flappy Bird game using reinforcement learning.

---

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [How It Works](#how-it-works)
- [Results](#results)

---

## 🎯 Overview

This project implements a **Deep Q-Network (DQN)** that learns to play Flappy Bird through trial and error.

The agent:
- Starts with random actions
- Learns from experience
- Improves over time using neural networks

**Goal:** Maximize score by successfully passing pipes.

---

## ✨ Features

- Deep Q-Learning (DQN)
- Epsilon-Greedy Strategy
- Experience Replay Memory
- Custom Flappy Bird Environment (Pygame)
- GPU Support (CUDA if available)
- Training Visualization
- Pre-trained Model included
- Easy-to-tune hyperparameters

---

## 🏗️ Architecture

### 1. Environment (`environment.py`)
- Simulates Flappy Bird
- Handles physics, collisions, scoring
- Returns state to agent

### 2. Agent (`agent.py`)
- Implements DQN algorithm
- Uses epsilon-greedy policy
- Learns from replay memory

### 3. Model (`model.py`)

Neural Network:

```

Input: 4 values
Hidden: 64 → 64
Output: 2 actions

```

State includes:
- Bird height
- Velocity
- Distance to pipe
- Pipe gap offset

### 4. Memory (`memory.py`)
- Stores experiences:
```

(state, action, reward, next_state, done)

```
- Enables stable learning via random sampling

### 5. Training (`train.py`)
- Runs episodes
- Trains the model
- Saves progress

### 6. Visualization (`plot.py`)
- Generates training graphs

---

## 📁 Project Structure

```

flappy_bird_ai/
├── agent.py
├── model.py
├── environment.py
├── memory.py
├── config.py
├── train.py
├── main.py
├── plot.py
├── model.pth
├── training_progress.png
└── README.md

````

---

## 🚀 Installation

### Requirements
- Python 3.8+
- pip

### Setup

```bash
git clone https://github.com/ahmedxx17/AI_GAME_TRAINER.git
cd AI_GAME_TRAINER
````

Create virtual environment:

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

Install dependencies:

```bash
pip install pygame numpy torch
```

---

## 💻 Usage

### Train the Agent

```bash
python flappy_bird_ai/train.py
```

This will:

* Train for multiple episodes
* Save model to `model.pth`
* Generate training graph

---

### Run Trained Model

python flappy_bird_ai/main.py

---

## ⚙️ Configuration

Edit `config.py`:

NUM_EPISODES = 2000
BATCH_SIZE = 32
LEARNING_RATE = 0.001
GAMMA = 0.99

EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995

MEMORY_CAPACITY = 10000

---

## 🧠 How It Works

1. Agent observes state
2. Chooses action (explore vs exploit)
3. Stores experience
4. Learns using replay memory
5. Updates Q-values using Bellman Equation:

Q(s, a) = r + γ * max(Q(s', a'))


---

## 📊 Results

* Early: random gameplay
* Mid: learning patterns
* Late: high scores and stable gameplay

---

## 🤝 Contributing

Feel free to:

* Improve model
* Add new RL techniques (Double DQN, etc.)
* Optimize performance

---

## 📝 License

MIT License

---

## 🙏 Acknowledgments

* Flappy Bird inspiration
* PyTorch
* Pygame
* DeepMind DQN research

```
