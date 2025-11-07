# RL Algorithms for RoboRacer

This directory contains CleanRL-style implementations of reinforcement learning algorithms optimized for the DonkeyEnv simulator with PufferLib vectorization.

## Overview

The implementations follow the CleanRL philosophy of:
- Single-file implementations that are easy to understand
- Minimal dependencies
- Clear documentation
- Reproducible results

All algorithms share common utilities through `rl_utils.py` to keep the code DRY (Don't Repeat Yourself).

## Files

### Core Implementations

- **`ppo_pufferlib.py`** - Proximal Policy Optimization (PPO)
  - On-policy algorithm
  - Best for: Sample efficiency with parallel environments
  - Based on: https://docs.cleanrl.dev/rl-algorithms/ppo/

- **`sac_pufferlib.py`** - Soft Actor-Critic (SAC)
  - Off-policy algorithm with replay buffer
  - Best for: Continuous control tasks, sample efficiency
  - Features: Automatic entropy tuning, double Q-learning
  - Based on: https://docs.cleanrl.dev/rl-algorithms/sac/

### Shared Utilities

- **`rl_utils.py`** - Common utilities shared across algorithms
  - `CNNFeatureExtractor`: Shared CNN architecture for visual observations
  - `layer_init`: Orthogonal weight initialization
  - `EpisodeMetricsLogger`: Episode metrics tracking and TensorBoard logging
  - `set_random_seeds`: Reproducibility utilities
  - `setup_logging_dirs`: Logging directory setup
  - `prepare_observation`: Observation preprocessing
  - `clip_action_to_space`: Action clipping utilities
  - `extract_episode_metrics`: Extract metrics from environment info

## Usage

### Training with PPO

```bash
# Basic training with default parameters
python src/algorithms/ppo_pufferlib.py

# Customize training parameters
python src/algorithms/ppo_pufferlib.py \
  --env-name donkey-circuit-launch-track-v0 \
  --num-envs 8 \
  --total-timesteps 2000000 \
  --learning-rate 3e-4 \
  --num-steps 2048
```

### Training with SAC

```bash
# Basic training with default parameters
python src/algorithms/sac_pufferlib.py

# Customize training parameters
python src/algorithms/sac_pufferlib.py \
  --env-name donkey-circuit-launch-track-v0 \
  --num-envs 1 \
  --total-timesteps 1000000 \
  --buffer-size 100000 \
  --batch-size 256 \
  --policy-lr 3e-4 \
  --q-lr 3e-4 \
  --autotune
```

## Algorithm Comparison

### PPO (Proximal Policy Optimization)

**Pros:**
- Works well with parallel environments
- Stable training
- Good for exploration
- On-policy (always learning from recent experience)

**Cons:**
- Requires more environment interactions
- Less sample efficient than off-policy methods

**Best for:**
- When you have many parallel simulators
- Initial experimentation
- Stable, reliable training

### SAC (Soft Actor-Critic)

**Pros:**
- Very sample efficient (uses replay buffer)
- Maximum entropy framework encourages exploration
- Off-policy (can reuse old experience)
- Good for continuous control

**Cons:**
- More hyperparameters to tune
- Typically uses fewer parallel environments
- Slightly more complex implementation

**Best for:**
- When environment interactions are expensive
- Fine-tuning and optimization
- Achieving best final performance

## Configuration

Both algorithms use dataclass-based configuration for easy customization:

### PPO Configuration
- `num_envs`: 4-16 (more is better)
- `num_steps`: 2048 (steps per environment per rollout)
- `learning_rate`: 3e-4
- `batch_size`: 64
- `num_minibatches`: 32
- `update_epochs`: 10

### SAC Configuration
- `num_envs`: 1-4 (fewer than PPO)
- `buffer_size`: 100000-1000000
- `learning_starts`: 5000-10000
- `batch_size`: 256
- `policy_lr`: 3e-4
- `q_lr`: 3e-4
- `autotune`: True (automatic entropy tuning)

## Monitoring Training

Both algorithms log to TensorBoard:

```bash
tensorboard --logdir ./output/tensorboard
```

### Logged Metrics

**Episode Performance:**
- Episodic return
- Episode length

**Driving Performance:**
- Cross-track error (CTE)
- Speed
- Forward velocity
- Collision rate

**Lap Performance:**
- Lap times (mean, min, std)
- Completed laps

**Training Metrics:**
- Loss values
- Learning rates
- Algorithm-specific metrics (KL divergence, entropy, Q-values, etc.)

## Model Checkpoints

Models are automatically saved to `./output/models/` with timestamps:
- PPO: `ppo_donkey_update{N}.pt` and `ppo_donkey_final.pt`
- SAC: `sac_donkey_step{N}.pt` and `sac_donkey_final.pt`

Each checkpoint contains:
- Model state dict(s)
- Optimizer state dict(s)
- Training progress (steps/updates)
- Algorithm-specific parameters

## Architecture

Both algorithms use a shared CNN architecture from `rl_utils.py`:

```
Input: HxWxC image (e.g., 120x160x3)
  ↓
Conv2D(32, 8x8, stride=4) + ReLU
  ↓
Conv2D(64, 4x4, stride=2) + ReLU
  ↓
Conv2D(64, 3x3, stride=1) + ReLU
  ↓
Flatten
  ↓
Feature Vector (input to actor/critic heads)
```

### PPO Network
- Shared CNN feature extractor
- Actor head: Linear(256) → Linear(action_dim)
- Critic head: Linear(256) → Linear(1)

### SAC Networks
- Actor: Shared CNN → Linear(256) → Gaussian policy with tanh squashing
- Q-Networks (x2): Shared CNN → Concat(features, action) → Linear(256) → Linear(256) → Linear(1)

## Extending

To add a new algorithm:

1. Create a new file (e.g., `td3_pufferlib.py`)
2. Import utilities from `rl_utils.py`:
   ```python
   from rl_utils import (
       CNNFeatureExtractor,
       EpisodeMetricsLogger,
       # ... other utilities
   )
   ```
3. Follow the structure of PPO or SAC
4. Update this README

## Requirements

See `requirements.txt` in the project root for dependencies.

Key dependencies:
- PyTorch
- NumPy
- TensorBoard
- PufferLib (for vectorization)
- gym-donkeycar

## References

- CleanRL: https://docs.cleanrl.dev/
- PPO Paper: https://arxiv.org/abs/1707.06347
- SAC Paper: https://arxiv.org/abs/1801.01290
- PufferLib: https://github.com/PufferAI/PufferLib

