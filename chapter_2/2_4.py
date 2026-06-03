from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


N_ARMS = 10
N_STEPS = 100
N_RUNS = 2000
EPSILON = 0.1
ALPHA = 0.1
RANDOM_WALK_STD = 0.01
REWARD_STD = 1.0
SEED = 24


def epsilon_greedy(q_estimates: np.ndarray, epsilon: float, rng: np.random.Generator) -> int:
    """Choose an action with epsilon-greedy action selection."""
    if rng.random() < epsilon:
        return int(rng.integers(len(q_estimates)))

    max_value = np.max(q_estimates)
    greedy_actions = np.flatnonzero(q_estimates == max_value)
    return int(rng.choice(greedy_actions))


def run_experiment(
    n_steps: int = N_STEPS,
    n_runs: int = N_RUNS,
    n_arms: int = N_ARMS,
    epsilon: float = EPSILON,
    alpha: float = ALPHA,
    random_walk_std: float = RANDOM_WALK_STD,
    reward_std: float = REWARD_STD,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Compare sample-average and constant-step-size estimates.

    The true action values start from the same value, zero, and then take an
    independent Gaussian random walk on every step.
    """
    rng = np.random.default_rng(seed)
    sample_average_q = np.zeros((n_runs, n_steps))
    constant_step_q = np.zeros((n_runs, n_steps))

    for run in range(n_runs):
        true_q = np.zeros(n_arms)
        sample_q = np.zeros(n_arms)
        constant_q = np.zeros(n_arms)
        sample_action_counts = np.zeros(n_arms)

        for step in range(n_steps):
            true_q += rng.normal(0.0, random_walk_std, size=n_arms)

            sample_action = epsilon_greedy(sample_q, epsilon, rng)
            sample_reward = rng.normal(true_q[sample_action], reward_std)
            sample_action_counts[sample_action] += 1
            sample_q[sample_action] += (
                sample_reward - sample_q[sample_action]
            ) / sample_action_counts[sample_action]
            sample_average_q[run, step] = sample_q[sample_action]

            constant_action = epsilon_greedy(constant_q, epsilon, rng)
            constant_reward = rng.normal(true_q[constant_action], reward_std)
            constant_q[constant_action] += alpha * (
                constant_reward - constant_q[constant_action]
            )
            constant_step_q[run, step] = constant_q[constant_action]

    return sample_average_q.mean(axis=0), constant_step_q.mean(axis=0)


def plot_results(
    sample_average_q: np.ndarray,
    constant_step_q: np.ndarray,
    output_path: Path | None = None,
) -> None:
    """Plot and save the action-value estimate traces."""
    steps = np.arange(1, len(sample_average_q) + 1)
    if output_path is None:
        output_path = Path(__file__).with_name("2_4.png")

    plt.figure(figsize=(9, 5))
    plt.plot(steps, sample_average_q, label="Sample average", color="tab:blue")
    plt.plot(
        steps,
        constant_step_q,
        label="Constant step-size alpha = 0.1",
        color="tab:orange",
    )
    plt.xlabel("Steps")
    plt.ylabel("Average estimated action value Q")
    plt.title("Exercise 2.4: Nonstationary 10-armed Bandit")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.show()


def main() -> None:
    sample_average_q, constant_step_q = run_experiment()
    plot_results(sample_average_q, constant_step_q)


if __name__ == "__main__":
    main()


