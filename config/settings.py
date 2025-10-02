"""
Configuration settings for Hellbot training and environments.
"""

from typing import Dict, List, Tuple

# Environment curriculum configuration
# TRAINING INTENSITY GUIDE:
# - Quick test (50k base): 500k total steps, ~1-2 hours
# - Balanced (200k base): 2M total steps, ~4-8 hours  
# - Intensive (500k base): 5M total steps, ~10-20 hours
# - Professional (1M+ base): 10M+ total steps, ~20+ hours

ENVIRONMENT_CURRICULUM = [
    # (env_name, difficulty, timesteps_multiplier, description)
    ("VizdoomBasic-v0", 1, 0.3, "Basic movement and shooting"),        # Reduced - simple env
    ("VizdoomCorridor-v0", 2, 0.6, "Navigate corridor with enemies"),  # Moderate
    ("VizdoomDefendCenter-v0", 2, 1.0, "Defend center position"),      # Standard
    ("VizdoomDefendLine-v0", 3, 1.2, "Defend line formation"),         # More complex
    ("VizdoomHealthGathering-v0", 2, 0.8, "Health gathering survival"), # Survival skills
    ("VizdoomMyWayHome-v0", 4, 1.8, "Complex navigation"),             # Navigation heavy
    ("VizdoomPredictPosition-v0", 3, 1.2, "Predict and shoot"),        # Aiming skills
    ("VizdoomTakeCover-v0", 4, 1.5, "Advanced tactics"),               # Tactical skills
    ("VizdoomDeathmatch-v0", 5, 3.0, "Full deathmatch combat"),        # Most important - 3x training
]

# Alternative curriculums for different training intensities
QUICK_CURRICULUM = [
    # Minimal training for fast testing (30k base = 300k total)
    ("VizdoomDefendCenter-v0", 2, 0.5, "Basic defense"),
    ("VizdoomCorridor-v0", 2, 0.7, "Basic navigation"),
    ("VizdoomDeathmatch-v0", 3, 2.0, "Basic combat"),
]

INTENSIVE_CURRICULUM = [
    # Extended training for best performance (500k base = 5M+ total)
    ("VizdoomBasic-v0", 1, 0.2, "Basic movement and shooting"),
    ("VizdoomCorridor-v0", 2, 0.5, "Navigate corridor with enemies"),
    ("VizdoomDefendCenter-v0", 2, 1.0, "Defend center position"),
    ("VizdoomDefendLine-v0", 3, 1.2, "Defend line formation"),
    ("VizdoomHealthGathering-v0", 2, 0.8, "Health gathering survival"),
    ("VizdoomMyWayHome-v0", 4, 2.0, "Complex navigation"),
    ("VizdoomPredictPosition-v0", 3, 1.5, "Predict and shoot"),
    ("VizdoomTakeCover-v0", 4, 2.0, "Advanced tactics"),
    ("VizdoomDeathmatch-v0", 5, 4.0, "Full deathmatch combat - primary focus"),
]

# Default PPO hyperparameters (conservative)
DEFAULT_PPO_PARAMS = {
    "learning_rate": 1e-5,
    "n_epochs": 40,
    "batch_size": 512,
    "n_steps": 1024,
    "gamma": 0.96,
    "gae_lambda": 0.95,
    "ent_coef": 0.2,
    "vf_coef": 0.75,
    "clip_range": 0.2,
    "max_grad_norm": 0.5,
}

# Optimized PPO hyperparameters (recommended)
OPTIMIZED_PPO_PARAMS = {
    "learning_rate": 3e-4,  # Standard RL learning rate
    "n_epochs": 10,         # Reduced to prevent overfitting
    "batch_size": 256,      # Smaller batches for better gradient estimates
    "n_steps": 2048,        # More steps per update for better sample efficiency
    "gamma": 0.995,         # Higher gamma for longer-term planning
    "gae_lambda": 0.95,     # Keep standard value
    "ent_coef": 0.01,       # Lower entropy for more focused actions
    "vf_coef": 0.5,         # Standard value coefficient
    "clip_range": 0.2,      # Standard PPO clip range
    "max_grad_norm": 0.5,   # Gradient clipping for stability
}

# Optuna hyperparameter search spaces
OPTUNA_SEARCH_SPACE = {
    "learning_rate": {"type": "float", "low": 1e-5, "high": 1e-3, "log": True},
    "n_epochs": {"type": "int", "low": 5, "high": 20},
    "batch_size": {"type": "categorical", "choices": [128, 256, 512, 1024]},
    "n_steps": {"type": "categorical", "choices": [1024, 2048, 4096]},
    "gamma": {"type": "float", "low": 0.9, "high": 0.999},
    "gae_lambda": {"type": "float", "low": 0.8, "high": 0.99},
    "ent_coef": {"type": "float", "low": 0.001, "high": 0.1, "log": True},
    "vf_coef": {"type": "float", "low": 0.25, "high": 1.0},
    "clip_range": {"type": "float", "low": 0.1, "high": 0.3},
    "max_grad_norm": {"type": "float", "low": 0.3, "high": 1.0},
}

# Training configuration
TRAINING_CONFIG = {
    "model_dir": "models/ppo",
    "model_filename": "hellbot.zip",
    "hyperparams_filename": "best_hyperparams.json",
    "n_parallel_envs": 8,
    "eval_episodes": 5,
    # Checkpoint configuration
    "checkpoint_frequency": 50000,  # Save checkpoint every N steps
    "max_checkpoints": 5,           # Keep last N checkpoints
    "progress_filename": "training_progress.json",
    "backup_on_resume": True,       # Create backup before resuming
}

# Environment configuration
ENVIRONMENT_CONFIG = {
    "screen_resolution": "RES_320X240",
    "screen_format": "RGB24",
    "ticrate": 20,
    "episode_timeout": 2000,
    "render_settings": {
        "window_visible": False,
        "render_hud": False,
        "render_crosshair": False,
        "render_decals": False,
        "render_particles": False,
        "render_corpses": False,
        "sound_enabled": False,
    }
}