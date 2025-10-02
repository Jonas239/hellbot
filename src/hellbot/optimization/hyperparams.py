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
    
    def load_best_hyperparams(self) -> Dict[str, Any]:
        """Load best hyperparameters from previous optimization"""
        if os.path.exists(self.hyperparams_file):
            try:
                with open(self.hyperparams_file, 'r') as f:
                    params = json.load(f)
                print(f"Loaded optimized hyperparameters from: {self.hyperparams_file}")
                return params
            except Exception as e:
                print(f"Failed to load hyperparameters: {e}")
        
        print("No optimized hyperparameters found. Using pre-optimized defaults.")
        return self.get_optimized_params()
    
    def create_objective_function(self, env_creator, trainer_class):
        """Create objective function for Optuna optimization"""
        
        def objective(trial):
            try:
                # Get suggested hyperparameters
                hyperparams = self.suggest_hyperparams_with_optuna(trial)
                
                # Create environment and trainer
                env = env_creator()
                trainer = trainer_class(env, hyperparams)
                
                # Short training for evaluation
                trainer.train(total_timesteps=10000)
                
                # Evaluate performance
                avg_reward = trainer.evaluate(n_episodes=5)
                
                # Clean up
                trainer.cleanup()
                
                return avg_reward
                
            except Exception as e:
                print(f"Trial failed: {e}")
                return -1000  # Large penalty for failed trials
        
        return objective