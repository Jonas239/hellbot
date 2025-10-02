"""
Hellbot - A reinforcement learning bot for VizDoom environments.

This package provides curriculum learning across multiple VizDoom environments
with hyperparameter optimization support using Optuna.
"""

__version__ = "0.1.0"

from .training.trainer import HellbotTrainer
from .environments.vizdoom_env import VizdoomEnvironmentManager
from .optimization.hyperparams import HyperparameterOptimizer

__all__ = [
    "HellbotTrainer",
    "VizdoomEnvironmentManager", 
    "HyperparameterOptimizer",
]