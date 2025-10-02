"""
Training logic for Hellbot reinforcement learning.
"""

import torch
import os
import sys
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecTransposeImage
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
from typing import Dict, Any, Optional
import numpy as np
import json
import time
import shutil
from datetime import datetime

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from config.settings import ENVIRONMENT_CURRICULUM, TRAINING_CONFIG
from src.hellbot.environments.vizdoom_env import VizdoomEnvironmentManager
from src.hellbot.optimization.hyperparams import HyperparameterOptimizer


class HellbotTrainer:
    """Main trainer class for Hellbot reinforcement learning"""
    
    def __init__(self, device: str = "auto"):
        self.device = self._setup_device(device)
        self.env_manager = VizdoomEnvironmentManager()
        self.optimizer = HyperparameterOptimizer()
        self.model = None
        self.vec_env = None
        self.training_progress = {}
        self.current_phase = 0
        self.total_steps_completed = 0
    
    def _setup_device(self, device: str) -> str:
        """Setup and return the appropriate device for training"""
        if device == "auto":
            if torch.backends.mps.is_available():
                device = "mps"
                print("Using Apple Metal (MPS)")
            elif torch.cuda.is_available():
                device = "cuda"
                print(f"CUDA available: {torch.cuda.get_device_name(0)}")
            else:
                device = "cpu"
        
        print(f"Using device: {device}")
        return device
    
    def initialize_model(self, env, hyperparams: Optional[Dict[str, Any]] = None) -> PPO:
        """Initialize PPO model with given or loaded hyperparameters"""
        
        if hyperparams is None:
            hyperparams = self.optimizer.load_best_hyperparams()
        
        model = PPO(
            "MultiInputPolicy",
            env,
            verbose=1,
            device=self.device,
            **hyperparams
        )
        
        print(f"Initialized PPO model with hyperparams: {hyperparams}")
        return model
    
    def save_training_progress(self, model_dir: str) -> None:
        """Save training progress to file"""
        progress_file = os.path.join(model_dir, TRAINING_CONFIG["progress_filename"])
        
        progress_data = {
            "current_phase": self.current_phase,
            "total_steps_completed": self.total_steps_completed,
            "completed_environments": self.training_progress,
            "timestamp": datetime.now().isoformat(),
            "device": self.device
        }
        
        os.makedirs(model_dir, exist_ok=True)
        with open(progress_file, 'w') as f:
            json.dump(progress_data, f, indent=2)
        
        print(f"Training progress saved to: {progress_file}")
    
    def load_training_progress(self, model_dir: str) -> bool:
        """Load training progress from file. Returns True if progress loaded."""
        progress_file = os.path.join(model_dir, TRAINING_CONFIG["progress_filename"])
        
        if not os.path.exists(progress_file):
            print("No training progress found. Starting from beginning.")
            return False
        
        try:
            with open(progress_file, 'r') as f:
                progress_data = json.load(f)
            
            self.current_phase = progress_data.get("current_phase", 0)
            self.total_steps_completed = progress_data.get("total_steps_completed", 0)
            self.training_progress = progress_data.get("completed_environments", {})
            
            print(f"Loaded training progress:")
            print(f"  Current phase: {self.current_phase}")
            print(f"  Total steps completed: {self.total_steps_completed:,}")
            print(f"  Completed environments: {len(self.training_progress)}")
            
            return True
            
        except Exception as e:
            print(f"Failed to load training progress: {e}")
            print("Starting from beginning.")
            return False
    
    def create_backup(self, model_path: str) -> None:
        """Create timestamped backup of existing model"""
        if os.path.exists(model_path):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = model_path.replace(".zip", f"_backup_{timestamp}.zip")
            shutil.copy2(model_path, backup_path)
            print(f"Created backup: {backup_path}")
    
    def cleanup_old_checkpoints(self, model_dir: str) -> None:
        """Keep only the most recent N checkpoints"""
        max_checkpoints = TRAINING_CONFIG["max_checkpoints"]
        checkpoint_pattern = "hellbot_checkpoint_"
        
        # Find all checkpoint files
        checkpoints = []
        for file in os.listdir(model_dir):
            if file.startswith(checkpoint_pattern) and file.endswith(".zip"):
                file_path = os.path.join(model_dir, file)
                checkpoints.append((file_path, os.path.getmtime(file_path)))
        
        # Sort by modification time (newest first)
        checkpoints.sort(key=lambda x: x[1], reverse=True)
        
        # Remove old checkpoints
        for file_path, _ in checkpoints[max_checkpoints:]:
            try:
                os.remove(file_path)
                print(f"Removed old checkpoint: {os.path.basename(file_path)}")
            except Exception as e:
                print(f"Failed to remove {file_path}: {e}")
    
    def create_checkpoint_callback(self, model_dir: str) -> CheckpointCallback:
        """Create callback for automatic checkpointing during training"""
        checkpoint_freq = TRAINING_CONFIG["checkpoint_frequency"]
        
        return CheckpointCallback(
            save_freq=checkpoint_freq,
            save_path=model_dir,
            name_prefix="hellbot_checkpoint",
            verbose=1
        )
    
    def create_vectorized_env(self, env_name: str, difficulty: int, 
                             n_envs: Optional[int] = None) -> SubprocVecEnv:
        """Create vectorized environment for training"""
        
        n_envs = n_envs or TRAINING_CONFIG["n_parallel_envs"]
        
        def make_env():
            return self.env_manager.create_environment(env_name, difficulty)
        
        # Create vectorized environment
        vec_env = SubprocVecEnv([make_env for _ in range(n_envs)])
        vec_env = VecTransposeImage(vec_env)
        
        return vec_env
    
    def train_single_environment(self, env_name: str, total_timesteps: int, 
                               difficulty: int = 2, model_path: Optional[str] = None,
                               resume: bool = True) -> None:
        """Train on a single environment with checkpointing support"""
        
        print(f"Training on {env_name} with difficulty {difficulty} for {total_timesteps:,} timesteps...")
        
        # Setup paths
        if model_path is None:
            model_dir = TRAINING_CONFIG["model_dir"]
            model_filename = TRAINING_CONFIG["model_filename"]
            model_path = os.path.join(model_dir, model_filename)
        
        model_dir = os.path.dirname(model_path)
        os.makedirs(model_dir, exist_ok=True)
        
        # Create backup if resuming and backup is enabled
        if resume and TRAINING_CONFIG.get("backup_on_resume", True):
            self.create_backup(model_path)
        
        # Create environment
        self.vec_env = self.create_vectorized_env(env_name, difficulty)
        
        # Load or create model
        if os.path.exists(model_path) and resume:
            print(f"Loading existing model from: {model_path}")
            try:
                self.model = PPO.load(model_path, env=self.vec_env, device=self.device)
                print("Successfully loaded existing model for continued training")
            except Exception as e:
                print(f"Failed to load model: {e}. Creating new model.")
                self.model = self.initialize_model(self.vec_env)
        else:
            print("Initializing a new model.")
            self.model = self.initialize_model(self.vec_env)
        
        # Setup callbacks
        callbacks = []
        
        # Checkpoint callback for automatic saving
        checkpoint_callback = self.create_checkpoint_callback(model_dir)
        callbacks.append(checkpoint_callback)
        
        # Combine callbacks
        callback_list = CallbackList(callbacks) if callbacks else None
        
        # Train with automatic checkpointing
        try:
            print(f"Starting training with automatic checkpoints every {TRAINING_CONFIG['checkpoint_frequency']:,} steps...")
            self.model.learn(
                total_timesteps=total_timesteps,
                callback=callback_list,
                progress_bar=True
            )
            print(f"Training completed successfully!")
            
        except KeyboardInterrupt:
            print("\n" + "="*60)
            print("Training interrupted by user!")
            print("Model and progress have been automatically saved.")
            print("You can resume training by running the same command.")
            print("="*60)
        except Exception as e:
            print(f"Training error: {e}")
            print("Saving current progress...")
        finally:
            # Save final model
            self.model.save(model_path)
            print(f"Final model saved at: {model_path}")
            
            # Clean up old checkpoints
            self.cleanup_old_checkpoints(model_dir)
            
            # Update training progress
            self.total_steps_completed += total_timesteps
            
            # Cleanup
            self.cleanup()
    
    def train_curriculum(self, base_timesteps: int = 50_000, resume: bool = True) -> None:
        """Train through curriculum of increasingly difficult environments with checkpoint support"""
        
        model_dir = TRAINING_CONFIG["model_dir"]
        model_filename = TRAINING_CONFIG["model_filename"]
        
        # Load previous progress if resuming
        if resume:
            progress_loaded = self.load_training_progress(model_dir)
            if progress_loaded:
                print("\n" + "="*60)
                print("RESUMING CURRICULUM TRAINING")
                print(f"Continuing from phase {self.current_phase + 1}")
                print("="*60)
            else:
                print("\n" + "="*60)
                print("STARTING NEW CURRICULUM TRAINING")
                print("="*60)
        else:
            print("\n" + "="*60)
            print("STARTING FRESH CURRICULUM TRAINING")
            print("(Previous progress will be ignored)")
            print("="*60)
            self.current_phase = 0
            self.training_progress = {}
            self.total_steps_completed = 0
        
        curriculum = ENVIRONMENT_CURRICULUM
        total_phases = len(curriculum)
        
        try:
            # Start from current phase (0 if new training)
            for i in range(self.current_phase, total_phases):
                env_name, difficulty, multiplier, description = curriculum[i]
                timesteps = int(base_timesteps * multiplier)
                
                print(f"\nPhase {i+1}/{total_phases}: {env_name}")
                print(f"Description: {description}")
                print(f"Difficulty: {difficulty}")
                print(f"Timesteps: {timesteps:,}")
                print(f"Total progress: {self.total_steps_completed:,} steps completed")
                print("-" * 40)
                
                # Check if this phase was already completed
                if env_name in self.training_progress:
                    print(f"Phase {i+1} already completed. Skipping...")
                    continue
                
                # Check if environment is available
                if not self.env_manager.check_environment_availability(env_name):
                    print(f"Environment {env_name} not available. Skipping...")
                    self.training_progress[env_name] = "skipped"
                    continue
                
                # Update current phase
                self.current_phase = i
                
                # Train on this environment
                self.train_single_environment(
                    env_name=env_name,
                    total_timesteps=timesteps,
                    difficulty=difficulty,
                    model_path=os.path.join(model_dir, model_filename),
                    resume=resume
                )
                
                # Mark phase as completed
                self.training_progress[env_name] = {
                    "completed": True,
                    "timesteps": timesteps,
                    "difficulty": difficulty,
                    "completed_at": datetime.now().isoformat()
                }
                
                # Save progress after each phase
                self.save_training_progress(model_dir)
                
                print(f"✓ Completed phase {i+1}/{total_phases}: {env_name}")
            
            # Training completed
            print("\n" + "=" * 60)
            print("🎉 CURRICULUM TRAINING COMPLETED! 🎉")
            print(f"Total training steps: {self.total_steps_completed:,}")
            print(f"Environments completed: {len([v for v in self.training_progress.values() if isinstance(v, dict) and v.get('completed')])}/{total_phases}")
            print("The bot has been trained across multiple environments with increasing difficulty.")
            print("=" * 60)
            
        except KeyboardInterrupt:
            print("\n" + "="*60)
            print("CURRICULUM TRAINING INTERRUPTED")
            print(f"Progress saved. You can resume from phase {self.current_phase + 1}")
            print(f"To resume: Run the same training command")
            print("="*60)
        except Exception as e:
            print(f"\nCurriculum training error: {e}")
            print("Progress has been saved. You can try to resume training.")
        finally:
            # Final progress save
            self.save_training_progress(model_dir)
    
    def optimize_hyperparameters(self, env_name: str = "VizdoomDefendCenter-v0", 
                                n_trials: int = 20, timeout: int = 3600) -> Dict[str, Any]:
        """Optimize hyperparameters for a specific environment"""
        
        def create_env():
            return self.create_vectorized_env(env_name, 2, 4)  # Smaller for optimization
        
        objective = self.optimizer.create_objective_function(
            create_env, 
            lambda env, hyperparams: ModelTrainer(env, hyperparams, self.device)
        )
        
        return self.optimizer.optimize_hyperparameters(objective, n_trials, timeout)
    
    def evaluate(self, env_name: str = "VizdoomDefendCenter-v0", 
                n_episodes: int = 5, model_path: Optional[str] = None) -> float:
        """Evaluate trained model performance"""
        
        if model_path is None:
            model_dir = TRAINING_CONFIG["model_dir"]
            model_filename = TRAINING_CONFIG["model_filename"]
            model_path = os.path.join(model_dir, model_filename)
        
        if not os.path.exists(model_path):
            print(f"No model found at {model_path}")
            return 0.0
        
        # Create evaluation environment
        env = self.env_manager.create_environment(env_name, 2)
        model = PPO.load(model_path)
        
        total_rewards = []
        
        for episode in range(n_episodes):
            obs = env.reset()[0] if isinstance(env.reset(), tuple) else env.reset()
            total_reward = 0
            done = False
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, _ = env.step(action)
                total_reward += reward
                done = done or truncated
            
            total_rewards.append(total_reward)
            print(f"Episode {episode + 1}: {total_reward}")
        
        avg_reward = np.mean(total_rewards)
        print(f"Average reward over {n_episodes} episodes: {avg_reward:.2f}")
        
        env.close()
        return avg_reward
    
    def show_progress(self) -> None:
        """Display current training progress"""
        model_dir = TRAINING_CONFIG["model_dir"]
        
        if not self.load_training_progress(model_dir):
            print("No training progress found.")
            return
        
        print("\n" + "="*60)
        print("TRAINING PROGRESS SUMMARY")
        print("="*60)
        print(f"Current phase: {self.current_phase + 1}")
        print(f"Total steps completed: {self.total_steps_completed:,}")
        print(f"Completed environments: {len(self.training_progress)}")
        
        print("\nEnvironment Status:")
        for i, (env_name, _, multiplier, description) in enumerate(ENVIRONMENT_CURRICULUM):
            status = "✓ Completed" if env_name in self.training_progress else "⏳ Pending"
            if env_name in self.training_progress and self.training_progress[env_name] == "skipped":
                status = "⏭️  Skipped"
            print(f"  {i+1:2d}. {env_name:25s} - {status}")
        print("="*60)
    
    def reset_progress(self) -> None:
        """Reset training progress (start fresh)"""
        model_dir = TRAINING_CONFIG["model_dir"]
        progress_file = os.path.join(model_dir, TRAINING_CONFIG["progress_filename"])
        
        if os.path.exists(progress_file):
            # Create backup of progress
            backup_file = progress_file.replace(".json", f"_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            shutil.copy2(progress_file, backup_file)
            os.remove(progress_file)
            print(f"Training progress reset. Backup saved to: {backup_file}")
        else:
            print("No training progress to reset.")
        
        # Reset internal state
        self.current_phase = 0
        self.training_progress = {}
        self.total_steps_completed = 0
    
    def list_checkpoints(self) -> None:
        """List available checkpoints"""
        model_dir = TRAINING_CONFIG["model_dir"]
        
        if not os.path.exists(model_dir):
            print("No model directory found.")
            return
        
        print("\n" + "="*60)
        print("AVAILABLE CHECKPOINTS")
        print("="*60)
        
        # Main model
        main_model = os.path.join(model_dir, TRAINING_CONFIG["model_filename"])
        if os.path.exists(main_model):
            size = os.path.getsize(main_model) / (1024*1024)  # MB
            mtime = datetime.fromtimestamp(os.path.getmtime(main_model))
            print(f"📦 Main model: {TRAINING_CONFIG['model_filename']}")
            print(f"   Size: {size:.1f} MB, Modified: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Checkpoints
        checkpoints = []
        for file in os.listdir(model_dir):
            if file.startswith("hellbot_checkpoint_") and file.endswith(".zip"):
                file_path = os.path.join(model_dir, file)
                size = os.path.getsize(file_path) / (1024*1024)  # MB
                mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
                checkpoints.append((file, size, mtime))
        
        if checkpoints:
            checkpoints.sort(key=lambda x: x[2], reverse=True)  # Sort by time (newest first)
            print(f"\n💾 Automatic checkpoints ({len(checkpoints)}):")
            for file, size, mtime in checkpoints:
                print(f"   {file}")
                print(f"   Size: {size:.1f} MB, Created: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            print("\n💾 No automatic checkpoints found.")
        
        # Progress file
        progress_file = os.path.join(model_dir, TRAINING_CONFIG["progress_filename"])
        if os.path.exists(progress_file):
            mtime = datetime.fromtimestamp(os.path.getmtime(progress_file))
            print(f"\n📊 Progress file: {TRAINING_CONFIG['progress_filename']}")
            print(f"   Last updated: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
        
        print("="*60)
    
    def cleanup(self) -> None:
        """Clean up resources"""
        if self.vec_env:
            self.vec_env.close()
            self.vec_env = None


class ModelTrainer:
    """Helper class for hyperparameter optimization"""
    
    def __init__(self, env, hyperparams: Dict[str, Any], device: str):
        self.env = env
        self.hyperparams = hyperparams
        self.device = device
        self.model = None
    
    def train(self, total_timesteps: int) -> None:
        """Train model for given timesteps"""
        self.model = PPO(
            "MultiInputPolicy",
            self.env,
            verbose=0,  # Quiet for optimization
            device=self.device,
            **self.hyperparams
        )
        self.model.learn(total_timesteps=total_timesteps)
    
    def evaluate(self, n_episodes: int = 5) -> float:
        """Evaluate model performance"""
        if not self.model:
            return -1000
        
        total_rewards = []
        
        for _ in range(n_episodes):
            obs = self.env.reset()
            total_reward = 0
            done = False
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, _ = self.env.step(action)
                total_reward += reward[0] if isinstance(reward, np.ndarray) else reward
                if done:
                    break
            
            total_rewards.append(total_reward)
        
        return np.mean(total_rewards)
    
    def cleanup(self) -> None:
        """Clean up resources"""
        if self.env:
            self.env.close()