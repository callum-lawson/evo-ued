import jax
import jax.numpy as jnp
from jaxued.utils import accumulate_rollout_stats


def negative_mean_reward(dones, rewards, incomplete_value=-jnp.inf):
    """Compute standardized negative mean reward per level over completed episodes.

    This mirrors the original implementation in `jaxued.utils` but is provided here
    as an overlay utility so this repo can depend on upstream `jaxued` for the rest.

    Args:
        dones: Boolean array of shape (T, N) indicating episode terminations.
        rewards: Array of shape (T, N) with per-step rewards.
        incomplete_value: Value to return for levels with no completed episodes.

    Returns:
        Array of shape (N,) with scores equal to the negative z-score of the
        time-averaged episode rewards per level; incomplete levels receive
        `incomplete_value`.
    """
    mean_rewards, _, episode_count = accumulate_rollout_stats(
        dones, rewards, time_average=True
    )
    valid_mask = episode_count > 0
    num_valid = valid_mask.sum()

    denom = jnp.maximum(num_valid, 1).astype(mean_rewards.dtype)
    masked_sum = (mean_rewards * valid_mask).sum()
    batch_mean = masked_sum / denom

    centered = mean_rewards - batch_mean
    var = ((centered**2) * valid_mask).sum() / denom
    std = jnp.sqrt(var + 1e-8)

    z_scores = centered / std
    scores = -z_scores
    return jnp.where(valid_mask, scores, incomplete_value)
