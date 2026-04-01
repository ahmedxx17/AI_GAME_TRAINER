"""
config.py - Central configuration file for all hyperparameters.

All training, exploration, memory, and display settings are defined here
so they can be easily tuned without modifying any other file.
"""

# ─── Training Hyperparameters ───────────────────────────────────────
NUM_EPISODES    = 2000     # Total number of training episodes
MAX_STEPS       = 10000     # Max frames per episode (safety cap to prevent infinite loops)
BATCH_SIZE      = 32        # Number of experiences sampled from memory per training step
LEARNING_RATE   = 0.001     # Adam optimizer learning rate
GAMMA           = 0.99      # Discount factor: how much the agent values future rewards
TARGET_UPDATE_FREQ = 1000   # How often to sync target network with policy network

# ─── Exploration Hyperparameters ────────────────────────────────────
EPSILON_START   = 1.0       # Initial exploration rate (100% random actions)
EPSILON_MIN     = 0.01      # Minimum exploration rate (1% random actions)
EPSILON_DECAY   = 0.965    # Multiplicative decay applied after each episode

# ─── Replay Memory ─────────────────────────────────────────────────
MEMORY_CAPACITY = 10000     # Maximum number of experiences stored in replay buffer

# ─── Model Architecture ─────────────────────────────────────────────
STATE_SIZE      = 6         # Number of state features
ACTION_SIZE     = 2         # Number of possible actions
HIDDEN_SIZE     = 64        # Hidden layer width

# ─── Environment Features ───────────────────────────────────────────
MULTI_PIPE      = True      # Whether to keep multiple pipes on screen
NUM_PIPES       = 2         # Number of active pipes when MULTI_PIPE is enabled
PIPE_SPACING    = 200       # Horizontal distance between consecutive pipes

# ─── Reward Shaping ─────────────────────────────────────────────────
REWARD_DEATH    = -1.0      # Penalty for crashing
REWARD_PASS     = 1.0       # Bonus for passing a pipe
REWARD_ALIVE    = 0.1       # Small reward for each surviving frame

# ─── Display Settings ──────────────────────────────────────────────
FPS             = 60        # Frames per second when rendering the game window
RENDER_EVERY    = 50        # Show the game window every N episodes during training
PRINT_EVERY     = 100       # Print training statistics every N episodes
