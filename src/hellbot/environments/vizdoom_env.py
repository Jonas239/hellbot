"""
VizDoom environment management and wrapping utilities.
"""

import gymnasium as gym
from gymnasium.spaces import Discrete, MultiDiscrete, Box
from vizdoom import ScreenResolution, ScreenFormat, DoomGame, Mode, Button
import vizdoom
from typing import Union
import os
import sys
import numpy as np

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from config.settings import ENVIRONMENT_CONFIG


class BasicVizdoomEnv(gym.Env):
    """Basic VizDoom Environment that can be configured for different scenarios"""
    
    def __init__(self, scenario_file='basic.cfg', render_mode=None):
        super().__init__()
        self.game = DoomGame()
        self.render_mode = render_mode
        
        # Set scenario file
        scenario_path = self._get_scenario_path(scenario_file)
        if os.path.exists(scenario_path):
            self.game.load_config(scenario_path)
        else:
            # Fallback basic configuration if scenario file not found
            self._setup_basic_config()
        
        # Configure rendering
        self.game.set_window_visible(render_mode is not None)
        self.game.set_mode(Mode.PLAYER)
        
        # Initialize game
        self.game.init()
        
        # Set up action and observation spaces
        n_buttons = self.game.get_available_buttons_size()
        if n_buttons == 0:
            n_buttons = 3  # Fallback: move left, move right, attack
        self.action_space = Discrete(n_buttons)
        
        # Observation space (screen buffer)
        h, w = self.game.get_screen_height(), self.game.get_screen_width()
        self.observation_space = Box(
            low=0, high=255, shape=(h, w, 3), dtype=np.uint8
        )
        
    def _get_scenario_path(self, scenario_file):
        """Get path to VizDoom scenario file"""
        # Try to find scenario in VizDoom installation
        vizdoom_path = os.path.dirname(vizdoom.__file__)
        scenarios_path = os.path.join(vizdoom_path, 'scenarios')
        return os.path.join(scenarios_path, scenario_file)
        
    def _setup_basic_config(self):
        """Setup basic configuration if no scenario file found"""
        self.game.set_screen_resolution(ScreenResolution.RES_640X480)
        self.game.set_screen_format(ScreenFormat.RGB24)
        self.game.set_render_hud(False)
        self.game.set_render_crosshair(False)
        self.game.set_render_weapon(True)
        self.game.set_render_decals(False)
        self.game.set_render_particles(False)
        
        # Add basic buttons
        self.game.add_available_button(Button.MOVE_LEFT)
        self.game.add_available_button(Button.MOVE_RIGHT)
        self.game.add_available_button(Button.ATTACK)
        
        # Set game variables
        self.game.set_doom_skill(2)
        
    def reset(self, seed=None, options=None):
        if seed is not None:
            self.game.set_seed(seed)
        self.game.new_episode()
        state = self.game.get_state()
        if state is not None:
            observation = state.screen_buffer.transpose(1, 2, 0)  # CHW to HWC
        else:
            observation = np.zeros(self.observation_space.shape, dtype=np.uint8)
        return observation, {}
        
    def step(self, action):
        # Convert action to VizDoom format
        n_buttons = self.game.get_available_buttons_size()
        if n_buttons == 0:
            n_buttons = 3
        actions = [0] * n_buttons
        if action < len(actions):
            actions[action] = 1
            
        reward = self.game.make_action(actions)
        done = self.game.is_episode_finished()
        
        if done:
            observation = np.zeros(self.observation_space.shape, dtype=np.uint8)
        else:
            state = self.game.get_state()
            observation = state.screen_buffer.transpose(1, 2, 0) if state else np.zeros(self.observation_space.shape, dtype=np.uint8)
            
        return observation, reward, done, False, {}
        
    def close(self):
        if hasattr(self, 'game'):
            self.game.close()


# Register basic VizDoom environments
try:
    # Register a few basic environments that we can create
    registry = gym.envs.registry
    if hasattr(registry, 'env_specs'):
        env_specs = registry.env_specs
    else:
        env_specs = registry
    
    if 'VizdoomBasic-v0' not in env_specs:
        gym.register(
            id='VizdoomBasic-v0',
            entry_point=lambda: BasicVizdoomEnv('basic.cfg')
        )
    
    if 'VizdoomCorridor-v0' not in env_specs:
        gym.register(
            id='VizdoomCorridor-v0',
            entry_point=lambda: BasicVizdoomEnv('deadly_corridor.cfg')
        )
    
    if 'VizdoomDefendCenter-v0' not in env_specs:
        gym.register(
            id='VizdoomDefendCenter-v0',
            entry_point=lambda: BasicVizdoomEnv('defend_the_center.cfg')
        )
        
    if 'VizdoomDefendLine-v0' not in env_specs:
        gym.register(
            id='VizdoomDefendLine-v0',
            entry_point=lambda: BasicVizdoomEnv('defend_the_line.cfg')
        )
        
    if 'VizdoomHealthGathering-v0' not in env_specs:
        gym.register(
            id='VizdoomHealthGathering-v0',
            entry_point=lambda: BasicVizdoomEnv('health_gathering.cfg')
        )
        
    if 'VizdoomMyWayHome-v0' not in env_specs:
        gym.register(
            id='VizdoomMyWayHome-v0',
            entry_point=lambda: BasicVizdoomEnv('my_way_home.cfg')
        )
        
    if 'VizdoomPredictPosition-v0' not in env_specs:
        gym.register(
            id='VizdoomPredictPosition-v0',
            entry_point=lambda: BasicVizdoomEnv('predict_position.cfg')
        )
        
    if 'VizdoomTakeCover-v0' not in env_specs:
        gym.register(
            id='VizdoomTakeCover-v0',
            entry_point=lambda: BasicVizdoomEnv('take_cover.cfg')
        )
        
    if 'VizdoomDeathmatch-v0' not in env_specs:
        gym.register(
            id='VizdoomDeathmatch-v0',
            entry_point=lambda: BasicVizdoomEnv('deathmatch.cfg')
        )
    
    print("✓ VizDoom environments registered successfully")
    
except Exception as e:
    print(f"Warning: Could not register VizDoom environments: {e}")
    print("VizDoom package may not be properly installed")


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