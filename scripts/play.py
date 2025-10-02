#!/usr/bin/env python3
"""
Play script for testing trained Hellbot models.

IMPORTANT: Uses SAME visual settings as training to avoid confusing the model!
- Same resolution (320x240)
- Same visual elements (no HUD, no crosshair, etc.)
- Only enables window visibility and sound for human viewing

Usage:
    python scripts/play.py                           # Default environment
    python scripts/play.py VizdoomCorridor-v0      # Specific environment
    python scripts/play.py ENV 10                   # Custom episode count
    python scripts/play.py ENV 10 /path/to/model   # Custom model path
    python scripts/play.py ENV 10 MODEL --unsafe   # Enable HUD/crosshair (may confuse model)
"""

import sys
import os

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from hellbot.environments.vizdoom_env import VizdoomEnvironmentManager
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from config.settings import TRAINING_CONFIG, PLAY_ENVIRONMENT_CONFIG, ENVIRONMENT_CONFIG


def playtest_model(model_path: str, env_name: str, max_episodes: int = 5, unsafe_mode: bool = False):
    """Playtest a trained PPO model in the specified VizDoom environment
    
    Args:
        model_path: Path to the trained model
        env_name: VizDoom environment name
        max_episodes: Number of episodes to play
        unsafe_mode: If True, enable HUD/crosshair (may confuse the model)
    """
    
    # Choose configuration based on mode
    if unsafe_mode:
        print("⚠️  WARNING: Using unsafe mode with HUD/crosshair - may confuse the model!")
        config = {
            **ENVIRONMENT_CONFIG,
            "render_settings": {
                **ENVIRONMENT_CONFIG["render_settings"],
                "window_visible": True,
                "render_hud": True,
                "render_crosshair": True,
                "sound_enabled": True,
            }
        }
    else:
        print("✓ Using safe mode - same visual setup as training")
        config = PLAY_ENVIRONMENT_CONFIG
    
    env_manager = VizdoomEnvironmentManager()
    
    def make_env():
        return env_manager.create_environment(env_name, difficulty=2, render_mode='human')
    
    env = DummyVecEnv([make_env])
    
    # Load model
    try:
        model = PPO.load(model_path)
        print(f"Loaded model from: {model_path}")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return
    
    total_rewards = []
    
    for episode in range(max_episodes):
        obs = env.reset()
        total_reward = 0
        done = False
        
        print(f"\nStarting episode {episode + 1}/{max_episodes}")
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            total_reward += reward
            env.render()
        
        print(f"Episode {episode + 1} Reward: {total_reward}")
        total_rewards.append(total_reward)
    
    avg_reward = sum(total_rewards) / len(total_rewards)
    print(f"\nAverage Reward over {max_episodes} episodes: {avg_reward:.2f}")
    env.close()


def main():
    """Main play function"""
    
    # Parse command line arguments
    env_name = sys.argv[1] if len(sys.argv) > 1 else "VizdoomDefendCenter-v0"
    max_episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    
    # Check for unsafe mode flag
    unsafe_mode = '--unsafe' in sys.argv
    if unsafe_mode:
        sys.argv.remove('--unsafe')  # Remove flag from args
    
    if len(sys.argv) > 3:
        model_path = sys.argv[3]
    else:
        model_dir = TRAINING_CONFIG["model_dir"]
        model_filename = TRAINING_CONFIG["model_filename"]
        model_path = os.path.join(model_dir, model_filename)
    
    # Convert to absolute path if relative
    if not os.path.isabs(model_path):
        model_path = os.path.join(os.path.dirname(__file__), '..', model_path)
    
    print(f"🎮 Hellbot Playtest")
    print(f"Model: {model_path}")
    print(f"Environment: {env_name}")
    print(f"Episodes: {max_episodes}")
    print(f"Mode: {'Unsafe (HUD/Crosshair)' if unsafe_mode else 'Safe (Training-identical)'}")
    print("-" * 50)
    
    if not os.path.exists(model_path):
        print(f"❌ Error: Model file not found at {model_path}")
        print("Please train the model first using: mise run train")
        return
    
    try:
        playtest_model(model_path, env_name, max_episodes, unsafe_mode)
    except Exception as e:
        print(f"❌ Error during playtest: {e}")
        print(f"Make sure the environment '{env_name}' is available")


if __name__ == "__main__":
    main()