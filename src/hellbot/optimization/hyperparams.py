"""
Hyperparameter optimization using Optuna.
"""

import os
import json
import numpy as np
from typing import Dict, Any, Optional
import sys

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from config.settings import (
    DEFAULT_PPO_PARAMS, OPTIMIZED_PPO_PARAMS, 
    OPTUNA_SEARCH_SPACE, TRAINING_CONFIG
)

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    optuna = None


class HyperparameterOptimizer:
    """Manages hyperparameter optimization using Optuna"""
    
    def __init__(self, model_dir: str = None):
        self.model_dir = model_dir or TRAINING_CONFIG["model_dir"]
        self.hyperparams_file = os.path.join(
            self.model_dir, 
            TRAINING_CONFIG["hyperparams_filename"]
        )
    
    def get_default_params(self) -> Dict[str, Any]:
        """Get conservative default parameters"""
        return DEFAULT_PPO_PARAMS.copy()
    
    def get_optimized_params(self) -> Dict[str, Any]:
        """Get pre-optimized parameters for VizDoom environments"""
        return OPTIMIZED_PPO_PARAMS.copy()
    
    def suggest_hyperparams_with_optuna(self, trial) -> Dict[str, Any]:
        """Suggest hyperparameters using Optuna trial"""
        if not OPTUNA_AVAILABLE:
            return self.get_optimized_params()
        
        params = {}
        search_space = OPTUNA_SEARCH_SPACE
        
        for param_name, config in search_space.items():
            if config["type"] == "float":
                params[param_name] = trial.suggest_float(
                    param_name, 
                    config["low"], 
                    config["high"],
                    log=config.get("log", False)
                )
            elif config["type"] == "int":
                params[param_name] = trial.suggest_int(
                    param_name,
                    config["low"],
                    config["high"]
                )
            elif config["type"] == "categorical":
                params[param_name] = trial.suggest_categorical(
                    param_name,
                    config["choices"]
                )
        
        return params
    
    def optimize_hyperparameters(self, objective_func, n_trials: int = 20, 
                                timeout: int = 3600) -> Dict[str, Any]:
        """Run hyperparameter optimization"""
        if not OPTUNA_AVAILABLE:
            print("Optuna not available. Install with: pip install optuna")
            print("Using pre-optimized hyperparameters instead.")
            return self.get_optimized_params()
        
        print(f"Starting hyperparameter optimization...")
        print(f"Trials: {n_trials}, Timeout: {timeout/60:.1f} minutes")
        
        try:
            # Create study
            study = optuna.create_study(direction='maximize')
            study.optimize(objective_func, n_trials=n_trials, timeout=timeout)
            
            print("Optimization completed!")
            print(f"Best trial: {study.best_trial.number}")
            print(f"Best value: {study.best_value:.4f}")
            print("Best params:")
            for key, value in study.best_params.items():
                print(f"  {key}: {value}")
            
            # Save best hyperparameters
            self.save_hyperparams(study.best_params)
            
            return study.best_params
            
        except Exception as e:
            print(f"Optimization failed: {e}")
            return self.get_optimized_params()
    
    def save_hyperparams(self, params: Dict[str, Any]) -> None:
        """Save hyperparameters to file"""
        os.makedirs(self.model_dir, exist_ok=True)
        
        with open(self.hyperparams_file, 'w') as f:
            json.dump(params, f, indent=2)
        
        print(f"Hyperparameters saved to: {self.hyperparams_file}")
    
    def load_best_hyperparams(self, force_type: str = "auto") -> Dict[str, Any]:
        """Load best hyperparameters from previous optimization
        
        Args:
            force_type: "auto" (default), "optimized", "default", or "saved"
        """
        
        if force_type == "default":
            print("🔧 Using DEFAULT hyperparameters (conservative)")
            return self.get_default_params()
        elif force_type == "optimized":
            print("🚀 Using pre-OPTIMIZED hyperparameters (recommended)")
            return self.get_optimized_params()
        elif force_type == "saved":
            if os.path.exists(self.hyperparams_file):
                try:
                    with open(self.hyperparams_file, 'r') as f:
                        params = json.load(f)
                    print(f"📁 Using SAVED hyperparameters from: {self.hyperparams_file}")
                    return params
                except Exception as e:
                    print(f"Failed to load saved hyperparameters: {e}")
                    print("Falling back to optimized parameters...")
                    return self.get_optimized_params()
            else:
                print("No saved hyperparameters found. Using optimized defaults.")
                return self.get_optimized_params()
        else:  # auto mode
            if os.path.exists(self.hyperparams_file):
                try:
                    with open(self.hyperparams_file, 'r') as f:
                        params = json.load(f)
                    print(f"📁 Loaded Optuna-optimized hyperparameters from: {self.hyperparams_file}")
                    return params
                except Exception as e:
                    print(f"Failed to load hyperparameters: {e}")
            
            print("🚀 No Optuna results found. Using pre-optimized defaults.")
            return self.get_optimized_params()
    
    def create_objective_function(self, env_creator, trainer_class, 
                                 eval_timesteps: int = 50000):
        """Create objective function for Optuna optimization
        
        Args:
            env_creator: Function to create environment
            trainer_class: Trainer class to use
            eval_timesteps: Number of timesteps for evaluation (default: 50k)
        """
        
        def objective(trial):
            try:
                print(f"\n🔬 Trial {trial.number}: Testing hyperparameters...")
                
                # Get suggested hyperparameters
                hyperparams = self.suggest_hyperparams_with_optuna(trial)
                
                # Print key parameters being tested
                print(f"  Learning rate: {hyperparams.get('learning_rate', 'N/A')}")
                print(f"  Batch size: {hyperparams.get('batch_size', 'N/A')}")
                print(f"  Entropy coef: {hyperparams.get('ent_coef', 'N/A')}")
                
                # Create environment and trainer
                env = env_creator()
                trainer = trainer_class(env, hyperparams)
                
                # Meaningful training duration (50k steps = ~10-15 minutes)
                print(f"  Training for {eval_timesteps:,} steps...")
                trainer.train(total_timesteps=eval_timesteps)
                
                # Evaluate performance over multiple episodes
                print(f"  Evaluating performance...")
                avg_reward = trainer.evaluate(n_episodes=10)  # More episodes for stable evaluation
                
                print(f"  ✓ Trial {trial.number} result: {avg_reward:.2f}")
                
                # Clean up
                trainer.cleanup()
                
                return avg_reward
                
            except Exception as e:
                print(f"  ✗ Trial {trial.number} failed: {e}")
                return -1000  # Large penalty for failed trials
        
        return objective