import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import poisson


# =============================================================================
# ENVIRONMENT CONFIGURATION
# =============================================================================
MAX_CARS = 20
MAX_MOVE_CARS = 5
RENTAL_CREDIT = 10
MOVE_CAR_COST = 2
GAMMA = 0.9

PARKING_LIMIT = 10
SECOND_PARKING_LOT_COST = 4

RENT_REQS_1 = 3
RENT_REQS_2 = 4
RETURNS_1 = 3
RETURNS_2 = 2

POISSON_BOUND = 11
THETA = 0.1
IMPROVEMENT_TOLERANCE = 1e-5

OUTPUT_DIR = Path(__file__).resolve().parent
ORIGINAL_OUTPUT_PATH = OUTPUT_DIR / "ex4.7_original_check.png"
MODIFIED_OUTPUT_PATH = OUTPUT_DIR / "ex4.7_modified_solution.png"


# =============================================================================
# POISSON PROBABILITY CACHING
# =============================================================================
poisson_cache = {}


def poisson_prob(n, lam):
    """Return the Poisson probability for n events given expected rate lam."""
    if (n, lam) not in poisson_cache:
        poisson_cache[(n, lam)] = poisson.pmf(n, lam)
    return poisson_cache[(n, lam)]


def moving_cost(action, modified_problem):
    """Return the overnight moving cost for an action."""
    if modified_problem and action > 0:
        return max(action - 1, 0) * MOVE_CAR_COST
    return abs(action) * MOVE_CAR_COST


def parking_cost(cars_1_after_move, cars_2_after_move, modified_problem):
    """Return the extra parking cost after overnight moves."""
    if not modified_problem:
        return 0

    cost = 0
    if cars_1_after_move > PARKING_LIMIT:
        cost += SECOND_PARKING_LOT_COST
    if cars_2_after_move > PARKING_LIMIT:
        cost += SECOND_PARKING_LOT_COST
    return cost


def build_dynamics():
    """Pre-compute transition probabilities and expected rewards for both problems."""
    print("Pre-computing environmental dynamics for original and modified problems...")

    actions = np.arange(-MAX_MOVE_CARS, MAX_MOVE_CARS + 1)
    num_states = MAX_CARS + 1

    transitions = np.zeros((num_states, num_states, len(actions), num_states, num_states))
    original_rewards = np.zeros((num_states, num_states, len(actions)))
    modified_rewards = np.zeros((num_states, num_states, len(actions)))

    for s1 in range(num_states):
        for s2 in range(num_states):
            for action_idx, action in enumerate(actions):
                if action > s1 or -action > s2:
                    continue

                cars_1_morning = int(min(s1 - action, MAX_CARS))
                cars_2_morning = int(min(s2 + action, MAX_CARS))

                for req_1 in range(POISSON_BOUND):
                    prob_req_1 = poisson_prob(req_1, RENT_REQS_1)
                    for req_2 in range(POISSON_BOUND):
                        prob_req_2 = poisson_prob(req_2, RENT_REQS_2)

                        rented_1 = min(cars_1_morning, req_1)
                        rented_2 = min(cars_2_morning, req_2)
                        rental_reward = (rented_1 + rented_2) * RENTAL_CREDIT

                        cars_1_after_rent = cars_1_morning - rented_1
                        cars_2_after_rent = cars_2_morning - rented_2

                        for ret_1 in range(POISSON_BOUND):
                            prob_ret_1 = poisson_prob(ret_1, RETURNS_1)
                            for ret_2 in range(POISSON_BOUND):
                                prob_ret_2 = poisson_prob(ret_2, RETURNS_2)
                                joint_prob = prob_req_1 * prob_req_2 * prob_ret_1 * prob_ret_2

                                next_s1 = min(cars_1_after_rent + ret_1, MAX_CARS)
                                next_s2 = min(cars_2_after_rent + ret_2, MAX_CARS)

                                transitions[s1, s2, action_idx, next_s1, next_s2] += joint_prob
                                original_rewards[s1, s2, action_idx] += joint_prob * rental_reward
                                modified_rewards[s1, s2, action_idx] += joint_prob * rental_reward

                original_rewards[s1, s2, action_idx] -= moving_cost(action, modified_problem=False)
                modified_rewards[s1, s2, action_idx] -= moving_cost(action, modified_problem=True)
                modified_rewards[s1, s2, action_idx] -= parking_cost(
                    cars_1_morning,
                    cars_2_morning,
                    modified_problem=True,
                )

    print("Pre-computation completed.\n")
    return actions, transitions, original_rewards, modified_rewards


def policy_iteration(actions, transitions, rewards, label):
    """Run policy iteration and return the final value function, policy, and policy history."""
    num_states = MAX_CARS + 1
    values = np.zeros((num_states, num_states))
    policy = np.zeros((num_states, num_states), dtype=int)
    policy_history = [policy.copy()]
    iteration = 0

    while True:
        print(f"--- {label}: Policy Iteration {iteration} ---")

        sweep = 0
        while True:
            delta = 0.0
            old_values = values.copy()

            for s1 in range(num_states):
                for s2 in range(num_states):
                    old_value = values[s1, s2]
                    action_idx = np.where(actions == policy[s1, s2])[0][0]
                    values[s1, s2] = rewards[s1, s2, action_idx] + GAMMA * np.sum(
                        transitions[s1, s2, action_idx] * old_values
                    )
                    delta = max(delta, abs(old_value - values[s1, s2]))

            sweep += 1
            if delta < THETA:
                print(f"Policy evaluation stabilized after {sweep} sweeps. Max delta: {delta:.4f}")
                break

        policy_stable = True
        for s1 in range(num_states):
            for s2 in range(num_states):
                old_action = policy[s1, s2]
                best_action = old_action
                old_action_idx = np.where(actions == old_action)[0][0]
                best_value = rewards[s1, s2, old_action_idx] + GAMMA * np.sum(
                    transitions[s1, s2, old_action_idx] * values
                )

                for action in actions:
                    if action > s1 or -action > s2:
                        continue

                    action_idx = np.where(actions == action)[0][0]
                    action_value = rewards[s1, s2, action_idx] + GAMMA * np.sum(
                        transitions[s1, s2, action_idx] * values
                    )

                    if action_value > best_value + IMPROVEMENT_TOLERANCE:
                        best_value = action_value
                        best_action = action

                policy[s1, s2] = best_action
                if old_action != best_action:
                    policy_stable = False

        iteration += 1
        if policy_stable:
            print(f"{label}: converged after {iteration} policy improvement steps.\n")
            break

        policy_history.append(policy.copy())

    return values, policy, policy_history


def plot_policy(ax, policy, title, show_parking_limit):
    """Plot a policy contour map."""
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("#Cars at second location")
    ax.set_ylabel("#Cars at first location")
    ax.set_xlim(0, MAX_CARS)
    ax.set_ylim(0, MAX_CARS)

    if np.all(policy == policy[0, 0]):
        ax.text(MAX_CARS / 2, MAX_CARS / 2, f"{policy[0, 0]}", ha="center", va="center", fontsize=14)
    else:
        contours = ax.contour(
            policy,
            colors="black",
            linewidths=0.5,
            levels=np.arange(-5, 6),
            extent=[0, MAX_CARS, 0, MAX_CARS],
        )
        ax.clabel(contours, inline=True, fontsize=8, fmt="%d")

    if show_parking_limit:
        ax.axhline(PARKING_LIMIT, color="gray", linestyle=":", linewidth=0.8)
        ax.axvline(PARKING_LIMIT, color="gray", linestyle=":", linewidth=0.8)


def plot_solution(policy_history, values, output_path, title, show_parking_limit):
    """Save a policy sequence plus final state-value function figure."""
    panel_count = len(policy_history) + 1
    cols = 3
    rows = math.ceil(panel_count / cols)
    fig = plt.figure(figsize=(5 * cols, 4.8 * rows))

    for idx, policy in enumerate(policy_history):
        ax = fig.add_subplot(rows, cols, idx + 1)
        plot_policy(ax, policy, f"$\\pi_{idx}$", show_parking_limit)

    ax_3d = fig.add_subplot(rows, cols, len(policy_history) + 1, projection="3d")
    x = np.arange(MAX_CARS + 1)
    y = np.arange(MAX_CARS + 1)
    x, y = np.meshgrid(x, y)
    ax_3d.plot_wireframe(x, y, values, color="black", linewidth=0.6)
    ax_3d.set_title(f"$v_{{\\pi_{len(policy_history) - 1}}}$", fontsize=14, pad=20)
    ax_3d.set_xlabel("#Cars at second location", labelpad=10)
    ax_3d.set_ylabel("#Cars at first location", labelpad=10)
    ax_3d.set_zlabel("Value ($)", labelpad=10)
    ax_3d.view_init(elev=30, azim=-55)

    for panel_idx in range(panel_count + 1, rows * cols + 1):
        ax = fig.add_subplot(rows, cols, panel_idx)
        ax.axis("off")

    fig.suptitle(title, fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def main():
    actions, transitions, original_rewards, modified_rewards = build_dynamics()

    original_values, _, original_history = policy_iteration(
        actions,
        transitions,
        original_rewards,
        label="Original problem check",
    )
    plot_solution(
        original_history,
        original_values,
        ORIGINAL_OUTPUT_PATH,
        "Exercise 4.7 Check: Original Jack's Car Rental",
        show_parking_limit=False,
    )

    modified_values, _, modified_history = policy_iteration(
        actions,
        transitions,
        modified_rewards,
        label="Exercise 4.7 modified problem",
    )
    plot_solution(
        modified_history,
        modified_values,
        MODIFIED_OUTPUT_PATH,
        "Exercise 4.7: Modified Jack's Car Rental",
        show_parking_limit=True,
    )


if __name__ == "__main__":
    main()
