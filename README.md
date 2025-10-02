# Hellbot 🤖

A reinforcement learning bot for VizDoom environments featuring curriculum learning and hyperparameter optimization.

## 🏗️ Project Structure

```
hellbot/
├── src/hellbot/                    # Main package
│   ├── environments/               # Environment management
│   │   └── vizdoom_env.py         # VizDoom wrapper & action space standardization
│   ├── training/                   # Training logic
│   │   └── trainer.py             # Main trainer class with curriculum learning
│   └── optimization/               # Hyperparameter optimization
│       └── hyperparams.py         # Optuna integration
├── scripts/                        # Entry point scripts
│   ├── train.py                   # Training script
│   └── play.py                    # Play/evaluation script
├── config/                         # Configuration
│   └── settings.py                # All configuration settings
├── models/ppo/                     # Saved models and hyperparameters
├── mise.toml                       # Development environment & tasks
└── pyproject.toml                  # Dependencies & build config
```

## 🚀 Quick Start

1. **Setup environment**:
   ```bash
   mise install  # Install Python 3.12 and uv
   mise run setup  # Install dependencies and setup
   ```

2. **Train the bot**:
   ```bash
   mise run train  # Curriculum training across multiple environments
   ```

3. **Play with trained model**:
   ```bash
   mise run play  # Watch the bot play
   ```

## 📋 Available Tasks

### Training
- `mise run train` - Full curriculum training (50k base timesteps)
- `mise run train-fast` - Quick training (25k base timesteps)
- `mise run train-intensive` - Extended training (100k base timesteps)
- `mise run train-single` - Single environment training

### Optimization
- `mise run optimize` - Hyperparameter optimization (20 trials, 1 hour)
- `mise run optimize-quick` - Quick optimization (10 trials, 30 minutes)
- `mise run optimize-intensive` - Thorough optimization (50 trials, 3 hours)

### Playing
- `mise run play` - Play in default environment
- `mise run play-env -- VizdoomCorridor-v0` - Play in specific environment
- `mise run play-deathmatch` - Play in hardest environment

### Utilities
- `mise run check-cuda` - Verify CUDA setup
- `mise run status` - Show project status
- `mise run clean` - Clean up generated files

## 🎯 Features

- **Curriculum Learning**: Progressive training across 9 VizDoom environments
- **Action Space Standardization**: Seamless training across different environments
- **Hyperparameter Optimization**: Optuna-powered PPO parameter tuning  
- **Modular Architecture**: Clean, maintainable code structure
- **Easy Development**: Mise-powered development environment