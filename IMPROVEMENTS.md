# AI Game Trainer

---

## 🔴 Priority 1 — Critical Fixes

### 1.1 Populate `requirements.txt`

**File:** `requirements.txt`  
**Problem:** The file is completely empty. Nobody can install dependencies from it.

```diff
+pygame>=2.5.0
+torch>=2.0.0
+numpy>=1.24.0
+matplotlib>=3.7.0
```

---

### 1.2 Add Target Network (DQN Stability)

**File:** `flappy_bird_ai/agent.py`  
**Problem:** The same network is used for both action selection and target computation. This creates a "moving target" problem — every training step shifts the targets, causing instability. This is the single biggest algorithmic improvement available.

**Changes needed in `__init__`:**

```diff
 class DQNAgent:
     def __init__(self):
         self.model = DQN().to(self.device)
+        self.target_model = DQN().to(self.device)
+        self.target_model.load_state_dict(self.model.state_dict())
+        self.target_model.eval()
+        self.train_step_count = 0
+        self.target_update_freq = 1000  # sync target network every N steps
```

**Changes needed in `train()`:**

```diff
     def train(self):
         ...
         with torch.no_grad():
-            max_next_q = self.model(next_states).max(1)[0]
+            max_next_q = self.target_model(next_states).max(1)[0]
             target_q = rewards + (1 - dones) * GAMMA * max_next_q
         ...
         self.optimizer.step()
+
+        self.train_step_count += 1
+        if self.train_step_count % self.target_update_freq == 0:
+            self.target_model.load_state_dict(self.model.state_dict())
```

**Also add to `config.py`:**

```diff
+TARGET_UPDATE_FREQ = 1000  # How often to sync target network with policy network
```

---

### 1.3 Use Huber Loss Instead of MSE

**File:** `flappy_bird_ai/agent.py` (line 55)  
**Problem:** MSE amplifies large errors (e.g., death penalty of -1.0), which can cause gradient explosions. Huber loss (SmoothL1Loss) clips large errors, making training more stable.

```diff
-        self.criterion = nn.MSELoss()
+        self.criterion = nn.SmoothL1Loss()  # Huber loss: robust to outlier rewards
```

---

## 🟡 Priority 2 — Performance Optimizations

### 2.1 Cache Font Objects (Rendering Performance)

**Files:** `flappy_bird_ai/environment.py` (line 220), `flappy_bird_ai/train.py` (line 106), `flappy_bird_ai/main.py` (line 89)  
**Problem:** `pygame.font.SysFont(None, 24)` is called **every single frame**. Font creation involves OS-level font lookup and is expensive. This should be created once and reused.

**Fix in `environment.py`:**

```diff
 class FlappyBirdEnv:
     def __init__(self):
         ...
+        self._font = None

     def render(self, screen):
         ...
-        font = pygame.font.SysFont(None, 24)
+        if self._font is None:
+            self._font = pygame.font.SysFont(None, 24)
+        font = self._font
```

**Fix in `train.py` — create font once before the episode loop:**

```diff
+    font = None  # Will be initialized on first render

     for episode in range(NUM_EPISODES):
         ...
             if should_render:
-                font = pygame.font.SysFont(None, 24)
+                if font is None:
+                    font = pygame.font.SysFont(None, 24)
```

**Fix in `main.py` — create font once before the game loop:**

```diff
+    font = pygame.font.SysFont(None, 24)  # Create once

     while running:
         ...
-            font = pygame.font.SysFont(None, 24)
-            ai_text = font.render("AI Playing", True, (100, 255, 100))
+            ai_text = font.render("AI Playing", True, (100, 255, 100))
```

---

### 2.2 Add Gradient Clipping

**File:** `flappy_bird_ai/agent.py` (line 148)  
**Problem:** During early training, error signals can be very large (random actions → frequent deaths). Without gradient clipping, these large gradients can destabilize the network.

```diff
         self.optimizer.zero_grad()
         loss.backward()
+        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
         self.optimizer.step()
```

---

### 2.3 Support Multiple Pipes on Screen

**File:** `flappy_bird_ai/environment.py`  
**Problem:** Only one pipe exists at a time. Real Flappy Bird shows 2-3 pipes simultaneously, making it more challenging.

**Approach:** Refactor `self.pipe_x` and `self.gap_y` from single values to a list of pipe dictionaries:

```python
self.pipes = [
    {"x": self.SCREEN_W, "gap_y": random.randint(...)},
    {"x": self.SCREEN_W + self.SCREEN_W // 2, "gap_y": random.randint(...)},
]
```

Then update `step()`, `_check_collision()`, `_get_state()`, and `render()` to iterate over the list. The state vector would need to include info about the nearest pipe (or the nearest N pipes).

**Effort:** ~1 hour. Consider making this optional via a `MULTI_PIPE` config flag.

---

## 🟢 Priority 3 — Code Quality & Refactoring

### 3.1 Add `__init__.py`

**File:** [NEW] `flappy_bird_ai/__init__.py`  
**Problem:** Without this file, `flappy_bird_ai` is not a proper Python package. This matters for imports if the project grows.

```python
"""Flappy Bird AI - Deep Q-Learning Agent package."""
```

---

### 3.2 Add Type Hints to All Functions

**Files:** All `.py` files  
**Problem:** No function signatures have type annotations. Adding them improves readability and enables IDE autocompletion & static analysis (mypy/pyright).

**Example for `memory.py`:**

```diff
+import numpy as np
+from typing import List, Tuple

-    def push(self, state, action, reward, next_state, done):
+    def push(self, state: np.ndarray, action: int, reward: float,
+             next_state: np.ndarray, done: float) -> None:

-    def sample(self, batch_size):
+    def sample(self, batch_size: int) -> List[Tuple[np.ndarray, int, float, np.ndarray, float]]:

-    def __len__(self):
+    def __len__(self) -> int:
```

**Example for `agent.py`:**

```diff
-    def choose_action(self, state):
+    def choose_action(self, state: np.ndarray) -> int:

-    def train(self):
+    def train(self) -> None:

-    def decay_epsilon(self):
+    def decay_epsilon(self) -> None:
```

---

### 3.3 Make Model Architecture Configurable

**Files:** `flappy_bird_ai/config.py`, `flappy_bird_ai/model.py`  
**Problem:** Layer sizes (4→64→64→2) are hardcoded. Moving them to config makes experimentation easy.

**Add to `config.py`:**

```diff
+# ─── Model Architecture ────────────────────────────────────────
+STATE_SIZE   = 4       # Number of state features
+ACTION_SIZE  = 2       # Number of possible actions
+HIDDEN_SIZE  = 64      # Hidden layer width
```

**Update `model.py`:**

```diff
+from config import STATE_SIZE, ACTION_SIZE, HIDDEN_SIZE

 class DQN(nn.Module):
-    def __init__(self):
+    def __init__(self, state_size=STATE_SIZE, action_size=ACTION_SIZE, hidden_size=HIDDEN_SIZE):
         super(DQN, self).__init__()
         self.network = nn.Sequential(
-            nn.Linear(4, 64),
+            nn.Linear(state_size, hidden_size),
             nn.ReLU(),
-            nn.Linear(64, 64),
+            nn.Linear(hidden_size, hidden_size),
             nn.ReLU(),
-            nn.Linear(64, 2)
+            nn.Linear(hidden_size, action_size)
         )
```

---

### 3.4 Enrich State Representation

**File:** `flappy_bird_ai/environment.py` (lines 151-175)  
**Problem:** The 4D state only includes distance to gap center. Adding distance to each pipe edge gives the network more direct information.

```diff
+        pipe_top = self.gap_y - self.PIPE_GAP // 2
+        pipe_bottom = self.gap_y + self.PIPE_GAP // 2

         state = np.array([
             self.bird_y / self.SCREEN_H,
             (self.bird_vel + 10) / 20,
             (self.pipe_x - self.BIRD_X) / self.SCREEN_W,
-            (self.bird_y - gap_center) / self.SCREEN_H
+            (self.bird_y - gap_center) / self.SCREEN_H,
+            (self.bird_y - pipe_top) / self.SCREEN_H,      # Distance to top pipe edge
+            (pipe_bottom - self.bird_y) / self.SCREEN_H,   # Distance to bottom pipe edge
         ], dtype=np.float32)
```

> **Note:** This changes state from 4D to 6D. You must also update `STATE_SIZE` in config and the model input size accordingly.

---

### 3.5 Extract Reward Values to `config.py`

**Files:** `flappy_bird_ai/config.py`, `flappy_bird_ai/environment.py` (lines 110-115)  
**Problem:** Reward values are hardcoded inside `step()`. Moving them to config centralizes tuning.

**Add to `config.py`:**

```diff
+# ─── Reward Shaping ────────────────────────────────────────────
+REWARD_DEATH    = -1.0      # Penalty for crashing
+REWARD_PASS     = 1.0       # Bonus for passing a pipe
+REWARD_ALIVE    = 0.1       # Small reward for each surviving frame
```

**Update `environment.py`:**

```diff
+from config import REWARD_DEATH, REWARD_PASS, REWARD_ALIVE

         if done:
-            reward = -1.0
+            reward = REWARD_DEATH
         elif pipe_passed:
-            reward = 1.0
+            reward = REWARD_PASS
         else:
-            reward = 0.1
+            reward = REWARD_ALIVE
```

---

## 🔵 Priority 4 — Feature Additions

### 4.1 Model Checkpointing During Training

**File:** `flappy_bird_ai/train.py`  
**Problem:** If training crashes at episode 1500, all progress is lost. Save the best model periodically.

```python
best_score = 0

for episode in range(NUM_EPISODES):
    ...
    scores.append(env.score)

    # Save best model checkpoint
    if env.score > best_score:
        best_score = env.score
        torch.save(agent.model.state_dict(), "best_model.pth")
        print(f"  New best score: {best_score} — model saved!")

    # Also save every 500 episodes as a safety checkpoint
    if episode % 500 == 0 and episode > 0:
        torch.save(agent.model.state_dict(), f"checkpoint_ep{episode}.pth")
```

---

### 4.2 Add `--resume` Flag for Continued Training

**Files:** `flappy_bird_ai/main.py`, `flappy_bird_ai/train.py`  
**Problem:** Training always starts from scratch. A `--resume` flag would load existing weights and continue.

**Add to `main.py`:**

```diff
 parser.add_argument(
     "--resume", action="store_true",
     help="Resume training from an existing model.pth"
 )
```

**Update training logic:**

```python
if args.train:
    if args.resume:
        agent.model.load_state_dict(torch.load("model.pth", map_location=agent.device))
        agent.epsilon = EPSILON_MIN  # Skip exploration phase on resume
        print("Resumed from model.pth")
```

---

### 4.3 Add TensorBoard / Logging Integration

**File:** `flappy_bird_ai/train.py`  
**Problem:** Training metrics are only printed to console. TensorBoard provides real-time interactive charts.

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("runs/flappy_bird")

for episode in range(NUM_EPISODES):
    ...
    writer.add_scalar("Score", env.score, episode)
    writer.add_scalar("Epsilon", agent.epsilon, episode)
    writer.add_scalar("Average Score (100)", avg, episode)

writer.close()
```

Then view with: `tensorboard --logdir=runs`

---

### 4.4 Add `--episodes N` CLI Argument

**File:** `flappy_bird_ai/main.py`  
**Problem:** To change episode count, you must edit `config.py`. A CLI override is more convenient.

```diff
 parser.add_argument(
     "--episodes", type=int, default=None,
     help="Override NUM_EPISODES from config.py"
 )
```

Then pass it through to `train_agent()`.

---

### 4.5 Update Root `README.md`

**File:** `README.md`  
**Problem:** Currently just `# AI_GAME_TRAINER`. Should contain at minimum a description and link to the detailed README.

````markdown
# AI Game Trainer

A collection of AI agents that learn to play games using Deep Reinforcement Learning.

## Games

### Flappy Bird

A DQN agent that learns to play Flappy Bird from scratch.
See [flappy_bird_ai/README.md](flappy_bird_ai/README.md) for full details.

## Quick Start

```bash
pip install pygame torch numpy matplotlib
cd flappy_bird_ai
python main.py --train
python main.py --play
```
````

```

---

## Summary Table

| # | Change | File(s) | Priority | Effort |
|---|--------|---------|----------|--------|
| 1.1 | Populate requirements.txt | `requirements.txt` | 🔴 Critical | 5 min |
| 1.2 | Add target network | `agent.py`, `config.py` | 🔴 Critical | 30 min |
| 1.3 | Huber loss | `agent.py` | 🔴 Critical | 2 min |
| 2.1 | Cache font objects | `environment.py`, `train.py`, `main.py` | 🟡 Perf | 10 min |
| 2.2 | Gradient clipping | `agent.py` | 🟡 Perf | 2 min |
| 2.3 | Multiple pipes | `environment.py` | 🟡 Feature | 1 hr |
| 3.1 | Add `__init__.py` | `flappy_bird_ai/__init__.py` | 🟢 Quality | 1 min |
| 3.2 | Type hints | All `.py` files | 🟢 Quality | 30 min |
| 3.3 | Configurable model arch | `config.py`, `model.py` | 🟢 Quality | 15 min |
| 3.4 | Richer state (6D) | `environment.py`, `config.py`, `model.py` | 🟢 Quality | 15 min |
| 3.5 | Extract rewards to config | `config.py`, `environment.py` | 🟢 Quality | 5 min |
| 4.1 | Model checkpointing | `train.py` | 🔵 Feature | 10 min |
| 4.2 | `--resume` flag | `main.py`, `train.py` | 🔵 Feature | 20 min |
| 4.3 | TensorBoard logging | `train.py` | 🔵 Feature | 30 min |
| 4.4 | `--episodes` CLI arg | `main.py` | 🔵 Feature | 5 min |
| 4.5 | Root README update | `README.md` | 🔵 Docs | 10 min |

**Total estimated effort: ~3.5 hours**
```
