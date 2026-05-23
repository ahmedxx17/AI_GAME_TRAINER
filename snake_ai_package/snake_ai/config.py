"""
config.py - Central configuration for the Snake CNN-DQN agent.

All training, exploration, memory, and display settings are defined here
so they can be easily tuned without modifying any other file.
"""

# ─── Training Hyperparameters ───────────────────────────────────────
NUM_EPISODES    = 5000      # Total training episodes
MAX_STEPS       = 500       # Max frames per episode (safety cap)
BATCH_SIZE      = 32        # Mini-batch size for gradient updates
LEARNING_RATE   = 0.0005    # Adam optimiser learning rate
GAMMA           = 0.99      # Discount factor for future rewards
TARGET_UPDATE_FREQ = 1000   # Sync target network every N training steps
TRAIN_START     = 1000      # Min experiences in memory before training begins
TRAIN_EVERY     = 2         # Perform one gradient step every N environment steps

# ─── Exploration Hyperparameters ────────────────────────────────────
EPSILON_START   = 1.0       # Initial exploration rate (100 % random)
EPSILON_MIN     = 0.01      # Floor exploration rate (1 % random)
EPSILON_DECAY   = 0.995     # Per-episode multiplicative decay

# ─── Replay Memory ─────────────────────────────────────────────────
MEMORY_CAPACITY = 10000     # Max stored transitions (uint8 frames keep RAM low)

# ─── CNN Input ──────────────────────────────────────────────────────
FRAME_SIZE      = 84        # Frames resized to 84×84 (DeepMind standard)
FRAME_STACK     = 4         # Consecutive frames stacked as CNN channels

# ─── Snake Game ─────────────────────────────────────────────────────
GRID_SIZE       = 10        # 10×10 cell grid
ACTION_SIZE     = 4         # Up / Right / Down / Left
HUNGER_LIMIT    = 200       # Steps without food before episode terminates

# ─── Reward Shaping ─────────────────────────────────────────────────
REWARD_FOOD     =  5.0      # Eating food
REWARD_DEATH    = -5.0      # Hitting wall or self
REWARD_CLOSER   =  0.5      # Moved closer to food (Manhattan distance)
REWARD_FARTHER  = -0.5      # Moved farther from food

# ─── Display Settings ──────────────────────────────────────────────
FPS             = 10        # Playback / render frame rate
DISPLAY_SIZE    = 400       # Pygame window size in pixels
RENDER_EVERY    = 100       # Show game window every N episodes during training
PRINT_EVERY     = 10        # Print stats every N episodes
