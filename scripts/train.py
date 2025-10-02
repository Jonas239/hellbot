#!/usr/bin/env python3
"""
Main training script for Hellbot.

Usage:
    python scripts/train.py                    # Curriculum training
    python scripts/train.py 100000           # Curriculum with custom base timesteps
    python scripts/train.py --single ENV     # Single environment training
    python scripts/train.py --optimize       # Hyperparameter optimization
"""

import sys
import os

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from hellbot.training.trainer import HellbotTrainer


def main():
    """Main training function with checkpoint management"""
    
    trainer = HellbotTrainer()
    
    if len(sys.argv) > 1 and sys.argv[1] == "--progress":
        # Show training progress
        trainer.show_progress()
        
    elif len(sys.argv) > 1 and sys.argv[1] == "--checkpoints":
        # List available checkpoints
        trainer.list_checkpoints()
        
    elif len(sys.argv) > 1 and sys.argv[1] == "--reset":
        # Reset training progress
        confirm = input("Are you sure you want to reset all training progress? (yes/no): ")
        if confirm.lower() == "yes":
            trainer.reset_progress()
            print("Training progress reset. Next training will start from scratch.")
        else:
            print("Reset cancelled.")
    
    elif len(sys.argv) > 1 and sys.argv[1] == "--fresh":
        # Start fresh training (ignore previous progress)
        base_timesteps = int(sys.argv[2]) if len(sys.argv) > 2 else 50_000
        print("Starting FRESH curriculum training (ignoring previous progress)")
        print(f"Base timesteps per environment: {base_timesteps:,}")
        
        trainer.train_curriculum(base_timesteps=base_timesteps, resume=False)
        
    elif len(sys.argv) > 1 and sys.argv[1] == "--optimize":
        # Hyperparameter optimization mode
        env_name = sys.argv[2] if len(sys.argv) > 2 else "VizdoomDefendCenter-v0"
        n_trials = int(sys.argv[3]) if len(sys.argv) > 3 else 20
        timeout = int(sys.argv[4]) if len(sys.argv) > 4 else 3600
        
        print("Starting hyperparameter optimization...")
        best_params = trainer.optimize_hyperparameters(env_name, n_trials, timeout)
        
    elif len(sys.argv) > 1 and sys.argv[1] == "--single":
        # Train on single environment
        env_name = sys.argv[2] if len(sys.argv) > 2 else "VizdoomDefendCenter-v0"
        total_timesteps = int(sys.argv[3]) if len(sys.argv) > 3 else 200_000
        difficulty = int(sys.argv[4]) if len(sys.argv) > 4 else 2
        resume = "--no-resume" not in sys.argv
        
        print(f"Training single environment: {env_name}")
        print(f"Resume mode: {'Enabled' if resume else 'Disabled'}")
        trainer.train_single_environment(
            env_name=env_name,
            total_timesteps=total_timesteps,
            difficulty=difficulty,
            resume=resume
        )
    else:
        # Default: Run curriculum training with resume
        base_timesteps = int(sys.argv[1]) if len(sys.argv) > 1 else 50_000
        print("Starting curriculum training across multiple VizDoom environments")
        print(f"Base timesteps per environment: {base_timesteps:,}")
        print("Resume mode: Enabled (use --fresh to start over)")
        
        trainer.train_curriculum(base_timesteps=base_timesteps, resume=True)


if __name__ == "__main__":
    main()