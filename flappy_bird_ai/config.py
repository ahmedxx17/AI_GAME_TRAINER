"""
config.py - Central configuration file for all hyperparameters.

All training, exploration, memory, and display settings are defined here
so they can be easily tuned without modifying any other file.
"""

# ─── Training Hyperparameters ───────────────────────────────────────
NUM_EPISODES    = 5000      # Total number of training episodes (was 2000 — too few)
MAX_STEPS       = 10000     # Max frames per episode (safety cap to prevent infinite loops)
BATCH_SIZE      = 128       # Samples per training step (was 32 — too noisy)
LEARNING_RATE   = 0.0005    # Adam LR (was 0.001 — caused oscillation; 5e-4 is DQN standard)
GAMMA           = 0.99      # Discount factor: how much the agent values future rewards
TARGET_UPDATE_FREQ = 500    # Sync target network every N steps (was 1000 — too slow to track)

# ─── Exploration Hyperparameters ────────────────────────────────────
EPSILON_START   = 1.0       # Initial exploration rate (100% random actions)
EPSILON_MIN     = 0.01      # Minimum exploration rate (1% random actions)
EPSILON_DECAY   = 0.9985    # Decay per episode — reaches ~0.01 around ep 3000 (was 0.995 → too fast/slow)

# ─── Replay Memory ─────────────────────────────────────────────────
MEMORY_CAPACITY = 50000     # Replay buffer size (was 10000 — got overwritten too quickly)

# ─── Model Architecture ─────────────────────────────────────────────
STATE_SIZE      = 6         # Number of state features
ACTION_SIZE     = 2         # Number of possible actions
HIDDEN_SIZE     = 128       # Hidden layer width (was 64 — too narrow for this task)

# ─── Environment Features ───────────────────────────────────────────
MULTI_PIPE      = True      # Whether to keep multiple pipes on screen
NUM_PIPES       = 2         # Number of active pipes when MULTI_PIPE is enabled
PIPE_SPACING    = 250       # Horizontal distance between consecutive pipes

# ─── Reward Shaping ─────────────────────────────────────────────────
REWARD_DEATH    = -1.0      # Penalty for crashing (aligned with reference impl)
REWARD_PASS     = 1.0       # Bonus for passing a pipe
REWARD_ALIVE    = 0.05      # Small reward for each surviving frame

# ─── Display Settings ──────────────────────────────────────────────
FPS             = 60        # Frames per second when rendering the game window
RENDER_EVERY    = 50        # Show the game window every N episodes during training
PRINT_EVERY     = 10       # Print training statistics every N episodes
