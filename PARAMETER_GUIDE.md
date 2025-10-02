# 🎯 Hellbot Parameter Guide

## **Current Status: You're Already Using Optimized Parameters!**

By default, Hellbot automatically uses **pre-optimized parameters** that work well for VizDoom environments. You don't need to do anything special!

## **Parameter Types Available**

### 🚀 **OPTIMIZED Parameters (Default - Recommended)**
```bash
mise run train                    # Uses optimized parameters automatically
mise run train-optimized-params  # Explicitly use optimized parameters
```
**What it is**: Pre-tuned parameters that work well for VizDoom environments
**Performance**: Good balance of speed and quality
**Use when**: This is your default choice - works great!

### 🔧 **DEFAULT Parameters (Conservative)**
```bash
mise run train-default-params
```
**What it is**: Conservative, safe parameters that always work
**Performance**: Slower learning but very stable
**Use when**: If optimized parameters cause issues or you want very stable training

### 📁 **SAVED Parameters (After Optuna)**
```bash
mise run train-saved-params       # Only works after running optimization
```
**What it is**: Parameters found by Optuna hyperparameter optimization
**Performance**: Best possible (if optimization worked well)
**Use when**: After you've run `mise run optimize` successfully

## **Hyperparameter Optimization (Optuna)**

### Why Trials Were Too Fast Before
- **Problem 1**: Only 10k training steps per trial (too short!)
- **Problem 2**: Wrong policy type causing crashes (`'Box' object has no attribute 'spaces'`)
- **Fixed**: Now uses 10k steps with proper CnnPolicy (~2-3 minutes each)
- **Result**: Reliable parameter evaluation without crashes

### Optimization Options

#### 🏃 **Quick Optimization** 
```bash
mise run optimize-quick    # 5 trials, ~15-20 minutes total
```

#### ⚖️ **Standard Optimization**
```bash
mise run optimize          # 20 trials, ~1-2 hours total, recommended
```

#### 🔥 **Intensive Optimization**
```bash
mise run optimize-intensive # 50 trials, ~3-4 hours total, best results
```

## **What's Actually Different?**

### Optimized vs Default Parameters:
```python
# DEFAULT (conservative)
learning_rate = 1e-4      # Slower learning
batch_size = 64           # Smaller batches
n_epochs = 4              # Fewer updates
ent_coef = 0.05          # More exploration

# OPTIMIZED (tuned for VizDoom)
learning_rate = 3e-4      # Faster learning
batch_size = 256          # Larger batches  
n_epochs = 10             # More updates
ent_coef = 0.01          # More focused actions
```

## **Recommendations**

### **For Normal Training:**
```bash
mise run train              # Just use this! Already optimized
```

### **If You Want the Absolute Best:**
1. Run optimization first:
   ```bash
   mise run optimize         # Takes ~5-6 hours
   ```

2. Then train with those results:
   ```bash
   mise run train-saved-params
   ```

### **If You Have Issues:**
```bash
mise run train-default-params  # Use conservative parameters
```

## **Current Parameter Values**

You can see the exact values in `config/settings.py`:
- `DEFAULT_PPO_PARAMS`: Conservative parameters
- `OPTIMIZED_PPO_PARAMS`: Pre-tuned parameters (used by default)
- `models/ppo/best_hyperparams.json`: Optuna results (if available)

## **TL;DR - What Should I Do?**

**Just run normal training - it's already optimized!**
```bash
mise run train
```

**If you want even better results and have time:**
```bash
mise run optimize        # Run once to find best parameters
mise run train-saved-params  # Then use those parameters
```