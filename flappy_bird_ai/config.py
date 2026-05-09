"""
config.py - Central configuration file for all hyperparameters.

All training, exploration, memory, and display settings are defined here
so they can be easily tuned without modifying any other file.
"""

# ─── Training Hyperparameters ───────────────────────────────────────
NUM_EPISODES    = 5000      # Total number of training episodes
MAX_STEPS       = 10000     # Max frames per episode (safety cap to prevent infinite loops)
BATCH_SIZE      = 64        # Stable batch size well-matched to replay buffer size
LEARNING_RATE   = 0.0003    # Fast enough to learn, stable enough to not forget
GAMMA           = 0.99      # Discount factor: how much the agent values future rewards
TARGET_UPDATE_FREQ = 500    # Frequent syncs stabilise learning

# ─── Exploration Hyperparameters ────────────────────────────────────
EPSILON_START   = 1.0       # Initial exploration rate (100% random actions)
EPSILON_MIN     = 0.01      # Minimum exploration rate (1% random actions)
EPSILON_DECAY   = 0.998     # Decays to ~0.10 by ep 2300, ~0.01 by ep 4600

# ─── Replay Memory ─────────────────────────────────────────────────
MEMORY_CAPACITY = 50000     # Large enough to be diverse, small enough to stay relevant

# ─── Model Architecture ─────────────────────────────────────────────
STATE_SIZE      = 6         # Number of state features
ACTION_SIZE     = 2         # Number of possible actions
HIDDEN_SIZE     = 128       # Right-sized for 6-input state space (not over/under capacity)

# ─── Environment Features ───────────────────────────────────────────
MULTI_PIPE      = True      # Whether to keep multiple pipes on screen
NUM_PIPES       = 2         # 2 pipes keeps state simple and learnable
PIPE_SPACING    = 280       # More spacing = more time to react between pipes

# ─── Reward Shaping ─────────────────────────────────────────────────
REWARD_DEATH    = -1.0      # Penalty for crashing
REWARD_PASS     = 1.0       # Bonus for passing a pipe
REWARD_ALIVE    = 0.01      # Tiny survival tick (does not drown out REWARD_PASS)

# ─── Display Settings ──────────────────────────────────────────────
FPS             = 60        # Frames per second when rendering the game window
RENDER_EVERY    = 50        # Show the game window every N episodes during training
PRINT_EVERY     = 1
       # Print stats every N episodes
