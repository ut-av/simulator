# Autonomous Driving DonkeySim

## Installation

If you haven't installed the uv package manager yet, install it: `curl -LsSf https://astral.sh/uv/install.sh | sh`

Then clone the repository, cd into it, and install dependencies with `uv sync`

## Run Training

### Quick Start

1. Start the simulator on port 9091
2. Run training:

```bash
# Single environment (for testing)
uv run python src/algorithms/ppo_pufferlib.py --num-envs 1
```

```bash
# Parallel training (faster, recommended)
# Note: You need to start multiple simulator instances on ports 9091-9094
uv run python src/algorithms/ppo_pufferlib.py --num-envs 4
```

### Tensorboard

```bash
uv run tensorboard --logdir ./output/tensorboard/
```


Then open http://localhost:6006

#### Logged Data

The training logs comprehensive metrics to TensorBoard organized into several categories. Understanding these metrics helps diagnose training progress and policy performance.

##### Episode Performance Metrics

**`charts/episodic_return`** - Total reward accumulated during an episode.
- **What it means**: Higher is better. Measures the cumulative reward the policy achieved.
- **Relation to policy**: Directly reflects policy quality. As the policy learns, this should increase and stabilize.
- **What to look for**: Steady upward trend indicates learning. Plateaus suggest convergence or need for hyperparameter tuning.

**`charts/episodic_length`** - Number of timesteps before episode termination.
- **What it means**: How long the car survives before crashing/going off track.
- **Relation to policy**: Early in training, episodes are short (frequent crashes). As the policy improves, episodes get longer.
- **What to look for**: Increasing length shows the policy is learning to avoid failures.

##### Driving Performance Metrics

**`driving/cross_track_error`** - Average absolute distance from track center (meters).
- **What it means**: How well the car stays centered on the track. Lower is better.
- **Relation to policy**: A good policy minimizes CTE while maintaining speed. Decreasing CTE indicates better track-following.
- **What to look for**: Values approaching 0 indicate precise center-line following. High CTE (near max_cte threshold) suggests the car is weaving or driving near track edges.

**`driving/cte_std`** - Standard deviation of cross track error.
- **What it means**: Consistency of lane positioning. Lower is better.
- **Relation to policy**: Low std means smooth, consistent driving. High std suggests erratic steering or oscillation.
- **What to look for**: Decreasing std over training indicates smoother control.

**`driving/speed`** - Average speed (units/sec).
- **What it means**: How fast the car is moving. The reward function typically encourages higher speed.
- **Relation to policy**: Initially, the policy may prioritize survival (low speed). As confidence grows, speed increases.
- **What to look for**: Speed should increase over training while maintaining low CTE (balance of speed vs safety).

**`driving/forward_velocity`** - Velocity component in the car's forward direction.
- **What it means**: Forward progress (can be negative if reversing). Differs from speed which is magnitude-only.
- **Relation to policy**: Negative values indicate the policy learned to reverse (possibly exploiting reward). Should be positive and increasing.
- **What to look for**: Should closely track speed for forward-driving policies.

**`driving/collision_rate`** - Proportion of timesteps with collisions (0.0 to 1.0).
- **What it means**: How often the car hits obstacles. Lower is better.
- **Relation to policy**: Well-trained policies avoid collisions. High rates indicate poor obstacle avoidance.
- **What to look for**: Should decrease toward 0 as training progresses.

##### Lap Performance Metrics

**`laps/lap_time_mean`** - Average time to complete a lap (seconds).
- **What it means**: Mean completion time across all completed laps in the rollout. Lower is better.
- **Relation to policy**: Ultimate performance metric for racing. Combines speed, efficiency, and control.
- **What to look for**: Decreasing lap times indicate the policy is getting faster. May initially be noisy (few completed laps), then stabilize.

**`laps/lap_time_min`** - Best (minimum) lap time achieved.
- **What it means**: The fastest lap completed. Lower is better.
- **Relation to policy**: Shows peak performance capability of the current policy.
- **What to look for**: This is your racing benchmark. Should decrease as training improves.

**`laps/lap_time_std`** - Standard deviation of lap times.
- **What it means**: Consistency across multiple laps. Lower is better.
- **Relation to policy**: Consistent policies produce similar lap times. High variance suggests unstable or stochastic behavior.
- **What to look for**: Low std indicates a reliable, consistent racing line.

**`laps/completed_laps`** - Total number of laps completed in the rollout period.
- **What it means**: How many full laps the agent(s) finished.
- **Relation to policy**: Early in training, may be 0 (can't complete a lap). Increases as policy improves.
- **What to look for**: Increasing count shows the policy can complete full circuits.

##### Training Loss Metrics

**`losses/policy_loss` (pg_loss)** - Policy gradient loss.
- **What it means**: Measures how much the policy is being updated. 
- **Relation to policy**: Large losses indicate big policy changes (exploration). Small losses suggest convergence.
- **What to look for**: Should be high initially (lots of learning), then decrease and stabilize. Sudden spikes may indicate instability.
- **Interpretation**: Not inherently "good" or "bad" - it's the gradient signal. What matters is whether episodic_return improves.

**`losses/value_loss` (v_loss)** - Value function (critic) loss.
- **What it means**: How accurately the critic predicts future rewards.
- **Relation to policy**: The critic guides the policy by estimating state values. High v_loss means poor value estimates, which hurts policy learning.
- **What to look for**: Should decrease over training as the critic learns to predict returns accurately. Plateauing v_loss with improving returns is ideal.

**`losses/entropy`** - Policy entropy (action randomness).
- **What it means**: Measures action diversity/exploration. Higher = more random, lower = more deterministic.
- **Relation to policy**: Early training needs high entropy (exploration). As the policy converges, entropy should decrease (more confident actions).
- **What to look for**: Gradual decrease from high to low. If entropy drops to zero too quickly, the policy may have converged prematurely. If it stays high, the policy remains too random.
- **Entropy coefficient** (`ent_coef=0.01`): Encourages exploration by penalizing low entropy. Tune this if entropy collapses or stays too high.

**`losses/approx_kl`** - Approximate KL divergence between old and new policies.
- **What it means**: How much the policy changed in one update. Larger values = bigger policy changes.
- **Relation to policy**: PPO constrains policy updates to prevent catastrophic changes. High KL means aggressive updates (risky). Low KL means conservative updates (safe but slow).
- **What to look for**: Should be moderate (e.g., < 0.05). If consistently high, reduce learning rate or increase clip_coef. If too low, policy barely updates.
- **Connection to clipping**: When KL is high, the clip mechanism activates to prevent excessive policy deviation.

**`losses/old_approx_kl`** - Alternative KL divergence estimate.
- **What it means**: Different approximation of KL divergence (less accurate but faster to compute).
- **Relation to policy**: Similar to approx_kl, used for monitoring policy update magnitude.
- **What to look for**: Should track approx_kl. Mostly useful for debugging.

**`losses/clipfrac`** - Fraction of training samples where PPO clipping was activated.
- **What it means**: How often the policy update was clipped (constrained). Range: 0.0 (never clipped) to 1.0 (always clipped).
- **Relation to policy**: High clipfrac means the policy wants to change a lot but is being constrained. Low clipfrac means the policy is updating freely within limits.
- **What to look for**: Moderate values (0.1-0.3) are typical. Very high clipfrac (>0.5) suggests the learning rate is too high or advantages are poorly scaled. Very low (<0.05) means the policy is barely constrained (could increase learning rate).

**`losses/explained_variance`** - How well the value function predicts returns.
- **What it means**: R² between predicted values and actual returns. Range: -∞ to 1.0. 1.0 = perfect predictions, 0.0 = no better than predicting the mean, negative = worse than predicting the mean.
- **Relation to policy**: Good value estimates are crucial for PPO. The critic's predictions inform advantage estimates, which guide the policy.
- **What to look for**: Should increase toward 1.0 as training progresses. Values near 1.0 indicate the critic accurately models the environment. Values near 0 or negative suggest the critic is struggling (bad sign).

**`charts/learning_rate`** - Current learning rate.
- **What it means**: Step size for gradient descent updates.
- **Relation to policy**: Controls update speed. Higher = faster learning but less stable. Lower = slower but more stable.
- **What to look for**: Typically constant unless using a scheduler. If manually tuning, reduce if training is unstable.

##### How Metrics Relate to Each Other

**Learning Progress**:
1. **Early Training**: High entropy (exploration), high v_loss (poor critic), short episodes, high CTE
2. **Mid Training**: Decreasing v_loss (critic improves), increasing episodic_length (survives longer), decreasing CTE (better control), moderate entropy
3. **Late Training**: High episodic_return (good rewards), low CTE (precise), high speed, low entropy (confident), stable losses, completing laps

**Diagnosing Issues**:
- **High episodic_return but high CTE**: Policy may be exploiting reward function in unintended ways (e.g., driving fast off-track)
- **Low v_loss but poor episodic_return**: Critic is accurate but policy isn't learning (check pg_loss, entropy, KL)
- **High clipfrac + high KL**: Learning rate too high, reduce it
- **Entropy → 0 too quickly**: Premature convergence, increase `ent_coef` or reduce learning rate
- **Completed laps = 0 for many updates**: Policy hasn't learned full track, may need more training or curriculum learning
- **Low explained_variance (<0.5)**: Critic is struggling, may need more training epochs or larger network

### (deprecated) Stable Baselines3 Implementation

```bash
uv run ./src/algorithms/ppo_sb3.py --visualize
```