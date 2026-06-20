from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


N_ARMS = 10
N_STEPS = 1000
N_RUNS = 500
EPSILON = 0.1
ALPHA = 0.1
RANDOM_WALK_STD = 0.01
SEED = 24


@dataclass(frozen=True)
class BanditConfig:
    n_steps: int = N_STEPS
    n_runs: int = N_RUNS
    n_arms: int = N_ARMS
    epsilon: float = EPSILON
    alpha: float = ALPHA
    random_walk_std: float = RANDOM_WALK_STD
    seed: int = SEED


def epsilon_greedy(q_estimates: np.ndarray, epsilon: float, rng: np.random.Generator) -> int:
    """Choose an action with epsilon-greedy action selection."""
    if rng.random() < epsilon:
        return int(rng.integers(len(q_estimates)))

    max_value = np.max(q_estimates)
    greedy_actions = np.flatnonzero(q_estimates == max_value)
    return int(rng.choice(greedy_actions))


def random_walk(
    q_a: np.ndarray,
    std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Move every action value by an independent Gaussian increment."""
    return q_a + rng.normal(0.0, std, size=q_a.shape)

def update_sample_average(
    q_estimates: np.ndarray,
    action_counts: np.ndarray,
    action: int,
    reward: float,
) -> None:
    action_counts[action] += 1
    q_estimates[action] += (reward - q_estimates[action]) / action_counts[action]


def update_constant_step_size(
    q_estimates: np.ndarray,
    action: int,
    reward: float,
    alpha: float,
) -> None:
    q_estimates[action] += alpha * (reward - q_estimates[action])


def run_experiment(
    n_steps: int = N_STEPS,
    n_runs: int = N_RUNS,
    n_arms: int = N_ARMS,
    epsilon: float = EPSILON,
    alpha: float = ALPHA,
    random_walk_std: float = RANDOM_WALK_STD,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Return sample-average and constant-step-size average reward curves."""
    config = BanditConfig(
        n_steps=n_steps,
        n_runs=n_runs,
        n_arms=n_arms,
        epsilon=epsilon,
        alpha=alpha,
        random_walk_std=random_walk_std,
        seed=seed,
    )
    rng = np.random.default_rng(seed)
    sample_average_rewards = np.zeros(config.n_steps)
    constant_step_rewards = np.zeros(config.n_steps)

    for _ in range(config.n_runs):
        q_a = np.zeros(config.n_arms)
        sample_average_q = np.zeros(config.n_arms)
        constant_step_q = np.zeros(config.n_arms)
        sample_action_counts = np.zeros(config.n_arms)

        for step in range(config.n_steps):
            q_a = random_walk(q_a, config.random_walk_std, rng)

            sample_action = epsilon_greedy(
                sample_average_q,
                config.epsilon,
                rng,
            )

            constant_action = epsilon_greedy(
                constant_step_q,
                config.epsilon,
                rng,
            )

            # Sample average
            sample_reward = q_a[sample_action]

            update_sample_average(
                sample_average_q,
                sample_action_counts,
                sample_action,
                sample_reward,
            )
            sample_average_rewards[step] += sample_reward

            # Constant step
            constant_reward = q_a[constant_action]

            update_constant_step_size(
                constant_step_q,
                constant_action,
                constant_reward,
                config.alpha,
            )
            constant_step_rewards[step] += constant_reward

    return sample_average_rewards / config.n_runs, constant_step_rewards / config.n_runs


def plot_results(
    sample_average_rewards: np.ndarray,
    constant_step_rewards: np.ndarray,
    output_path: Path | None = None,
) -> None:
    """Plot and save the average reward curves."""
    steps = np.arange(1, len(sample_average_rewards) + 1)
    if output_path is None:
        output_path = Path(__file__).with_name("ex2.4.png")

    plt.figure(figsize=(9, 5))
    plt.plot(steps, sample_average_rewards, label="Sample average", color="tab:red")
    plt.plot(
        steps,
        constant_step_rewards,
        label="Constant step-size",
        color="tab:blue",
    )
    plt.xlabel("Steps")
    plt.ylabel("Average reward")
    plt.title("Exercise 2.4: 10-armed Bandit")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.show()


def main() -> None:
    sample_average_rewards, constant_step_rewards = run_experiment()
    plot_results(sample_average_rewards, constant_step_rewards)


if __name__ == "__main__":
    main()

