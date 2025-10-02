"""
VizDoom environment management and wrapping utilities.
"""

import gymnasium as gym
from gymnasium.spaces import Discrete, MultiDiscrete
from vizdoom import ScreenResolution, ScreenFormat
from typing import Union
import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from config.settings import ENVIRONMENT_CONFIG


class ActionSpaceWrapper(gym.ActionWrapper):
    """Wrapper to standardize action spaces across different VizDoom environments"""
    
    def __init__(self, env, target_action_space):
        super().__init__(env)
        self.original_action_space = env.action_space
        self.target_action_space = target_action_space
        self.action_space = target_action_space
        
    def action(self, action):
        """Convert standardized action to environment-specific action"""
        if isinstance(self.original_action_space, Discrete):
            if isinstance(self.target_action_space, Discrete):
                # Clamp to valid range for original environment
                return min(action, self.original_action_space.n - 1)
            elif isinstance(self.target_action_space, MultiDiscrete):
                # Take first action from multi-discrete
                return min(action[0], self.original_action_space.n - 1)
        elif isinstance(self.original_action_space, MultiDiscrete):
            if isinstance(self.target_action_space, Discrete):
                # Convert single action to multi-discrete (duplicate action)
                return [min(action, max_val - 1) for max_val in self.original_action_space.nvec]
            elif isinstance(self.target_action_space, MultiDiscrete):
                # Map actions element-wise
                return [min(a, orig_max - 1) for a, orig_max in 
                       zip(action, self.original_action_space.nvec)]
        return action


class VizdoomEnvironmentManager:
    """Manages VizDoom environment creation and configuration"""
    
    def __init__(self):
        self.target_action_space = self._get_standardized_action_space()
    
    def _get_standardized_action_space(self) -> Discrete:
        """Get the standardized action space to use across all environments"""
        max_discrete = 8  # Most VizDoom envs have <= 8 actions
        return Discrete(max_discrete)
    
    def create_environment(self, env_name: str, difficulty: int = 2, 
                          render_mode: Union[str, None] = None) -> gym.Env:
        """Create a properly configured and wrapped VizDoom environment"""
        try:
            # Create base environment
            if render_mode:
                env = gym.make(env_name, render_mode=render_mode)
            else:
                env = gym.make(env_name)
            
            # Configure VizDoom settings
            if hasattr(env.unwrapped, 'game'):
                self._configure_vizdoom_game(env, difficulty, render_mode is not None)
            
            # Wrap with action space standardization
            env = ActionSpaceWrapper(env, self.target_action_space)
            return env
            
        except Exception as e:
            print(f"Failed to create environment {env_name}: {e}")
            return None
    
    def _configure_vizdoom_game(self, env: gym.Env, difficulty: int, 
                               is_rendering: bool = False) -> None:
        """Configure VizDoom game settings"""
        game = env.unwrapped.game
        config = ENVIRONMENT_CONFIG
        
        # Basic settings
        game.set_doom_skill(difficulty)
        game.set_ticrate(config["ticrate"])
        game.set_episode_timeout(config["episode_timeout"])
        
        # Screen settings
        screen_res = getattr(ScreenResolution, config["screen_resolution"])
        screen_fmt = getattr(ScreenFormat, config["screen_format"])
        game.set_screen_resolution(screen_res)
        game.set_screen_format(screen_fmt)
        
        # Render settings (different for training vs playing)
        render_settings = config["render_settings"]
        if is_rendering:
            # Enable HUD and crosshair for better visualization during play
            game.set_window_visible(True)
            game.set_render_hud(True)
            game.set_render_crosshair(True)
            game.set_screen_resolution(ScreenResolution.RES_800X600)  # Higher res for play
        else:
            # Minimal rendering for training speed
            game.set_window_visible(render_settings["window_visible"])
            game.set_render_hud(render_settings["render_hud"])
            game.set_render_crosshair(render_settings["render_crosshair"])
        
        # Always disable these for performance
        game.set_render_decals(render_settings["render_decals"])
        game.set_render_particles(render_settings["render_particles"])
        game.set_render_corpses(render_settings["render_corpses"])
        game.set_sound_enabled(render_settings["sound_enabled"])
    
    def get_available_environments(self) -> list:
        """Get list of available VizDoom environments"""
        envs = [env_id for env_id in gym.envs.registry.env_specs.keys() 
                if 'Vizdoom' in env_id]
        return sorted(envs)
    
    def check_environment_availability(self, env_name: str) -> bool:
        """Check if a specific environment is available"""
        try:
            test_env = gym.make(env_name)
            test_env.close()
            return True
        except Exception:
            return False