import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import poisson

# =============================================================================
# ENVIRONMENT CONFIGURATIONS & HYPERPARAMETERS
# =============================================================================
MAX_CARS = 20            # Maximum number of cars each location can hold
MAX_MOVE_CARS = 5        # Maximum number of cars that can be moved overnight
RENTAL_CREDIT = 10       # Reward per car rented out
MOVE_CAR_COST = 2        # Cost per car moved overnight
GAMMA = 0.9              # Discount factor for future rewards

# Expected values (lambda) for Poisson distributions
RENT_REQS_1 = 3
RENT_REQS_2 = 4
RETURNS_1 = 3
RETURNS_2 = 2

# Truncation bound for Poisson distributions to accelerate calculations
POISSON_BOUND = 11

# =============================================================================
# POISSON PROBABILITY CACHING
# =============================================================================
poisson_cache = {}

def poisson_prob(n, lam):
    """Returns the Poisson probability for n events given expected rate lam."""
    if (n, lam) not in poisson_cache:
        poisson_cache[(n, lam)] = poisson.pmf(n, lam)
    return poisson_cache[(n, lam)]

# =============================================================================
# PRE-COMPUTING TRANSITION DYNAMICS
# =============================================================================
# Pre-calculating transitions allows policy evaluation to run substantially faster.
# P_matrix stores the probability of ending up in next_state given current_state and action.
# R_matrix stores the expected reward given current_state and action.
print("Pre-computing environmental dynamics (this may take a moment)...")

# Valid actions range from -5 (move 5 from loc 2 to 1) to +5 (move 5 from loc 1 to 2)
actions = np.arange(-MAX_MOVE_CARS, MAX_MOVE_CARS + 1)
num_states = MAX_CARS + 1

P_matrix = np.zeros((num_states, num_states, len(actions), num_states, num_states))
R_matrix = np.zeros((num_states, num_states, len(actions)))

for s1 in range(num_states):
    for s2 in range(num_states):
        for a_idx, act in enumerate(actions):
            # Check if the moving action is physically feasible
            if act > 0 and s1 < act:
                continue  # Cannot move more cars than available at location 1
            if act < 0 and s2 < abs(act):
                continue  # Cannot move more cars than available at location 2
                
            # State configuration immediately after overnight transfers
            cars_1_morning = int(min(s1 - act, MAX_CARS))
            cars_2_morning = int(min(s2 + act, MAX_CARS))
            
            # Absolute immediate moving cost incurred overnight
            base_reward = -abs(act) * MOVE_CAR_COST
            
            # Joint loop over all possible rental requests next morning
            for req_1 in range(POISSON_BOUND):
                prob_req_1 = poisson_prob(req_1, RENT_REQS_1)
                for req_2 in range(POISSON_BOUND):
                    prob_req_2 = poisson_prob(req_2, RENT_REQS_2)
                    
                    # Actual cars rented cannot exceed inventory
                    valid_rent_1 = min(cars_1_morning, req_1)
                    valid_rent_2 = min(cars_2_morning, req_2)
                    
                    rent_reward = (valid_rent_1 + valid_rent_2) * RENTAL_CREDIT
                    
                    # Stock remaining after rentals are processed
                    cars_1_after_rent = cars_1_morning - valid_rent_1
                    cars_2_after_rent = cars_2_morning - valid_rent_2
                    
                    # Joint loop over all possible car returns in the evening
                    for ret_1 in range(POISSON_BOUND):
                        prob_ret_1 = poisson_prob(ret_1, RETURNS_1)
                        for ret_2 in range(POISSON_BOUND):
                            prob_ret_2 = poisson_prob(ret_2, RETURNS_2)
                            
                            # Final state at day's end capped by location capacity
                            s1_next = min(cars_1_after_rent + ret_1, MAX_CARS)
                            s2_next = min(cars_2_after_rent + ret_2, MAX_CARS)
                            
                            # Combined probability of this specific scenario
                            joint_prob = prob_req_1 * prob_req_2 * prob_ret_1 * prob_ret_2
                            
                            # Accumulate transitions and rewards
                            P_matrix[s1, s2, a_idx, s1_next, s2_next] += joint_prob
                            R_matrix[s1, s2, a_idx] += joint_prob * rent_reward
            
            # Incorporate the static overnight moving cost into the expected reward
            R_matrix[s1, s2, a_idx] += base_reward

print("Pre-computation completed successfully.\n")

# =============================================================================
# POLICY ITERATION ALGORITHM
# =============================================================================
# Initial state-value function and policy map setup
V = np.zeros((num_states, num_states))
policy = np.zeros((num_states, num_states), dtype=int)  # Initial policy moves 0 cars everywhere

policy_history = [policy.copy()]  # Record policies over iterations to replicate Fig 4.2
theta = 0.1                       # Policy evaluation accuracy threshold
iteration = 0

while True:
    print(f"--- Launching Policy Iteration {iteration} ---")
    
    # --- Step 1: Policy Evaluation ---
    eval_step = 0
    while True:
        delta = 0.0
        V_old = V.copy()  # Synchronous evaluation update backup
        for s1 in range(num_states):
            for s2 in range(num_states):
                old_v = V[s1, s2]
                act = policy[s1, s2]
                a_idx = np.where(actions == act)[0][0]
                
                # Bellman Expectation Equation using the pre-computed matrices
                V[s1, s2] = R_matrix[s1, s2, a_idx] + GAMMA * np.sum(P_matrix[s1, s2, a_idx] * V_old)
                delta = max(delta, abs(old_v - V[s1, s2]))
        
        eval_step += 1
        if delta < theta:
            print(f"Policy Evaluation stabilized after {eval_step} sweeps. Max Delta: {delta:.4f}")
            break
            
    # --- Step 2: Policy Improvement ---
    policy_stable = True
    for s1 in range(num_states):
        for s2 in range(num_states):
            old_action = policy[s1, s2]
            
            best_action = old_action
            # Resolve Exercise 4.4 bug: baseline value evaluation against current action
            old_a_idx = np.where(actions == old_action)[0][0]
            best_value = R_matrix[s1, s2, old_a_idx] + GAMMA * np.sum(P_matrix[s1, s2, old_a_idx] * V)
            
            # Evaluate all possible alternative relocation actions
            for act in actions:
                if act > s1 or -act > s2:
                    continue  # Filter out physically impossible actions
                    
                a_idx = np.where(actions == act)[0][0]
                action_value = R_matrix[s1, s2, a_idx] + GAMMA * np.sum(P_matrix[s1, s2, a_idx] * V)
                
                # Strict superiority evaluation condition to guarantee convergence
                if action_value > best_value + 1e-5:
                    best_value = action_value
                    best_action = act
            
            policy[s1, s2] = best_action
            if old_action != policy[s1, s2]:
                policy_stable = False
                
    iteration += 1
    
    if policy_stable:
        print(f"\nPolicy Iteration converged! Total iterations: {iteration}")
        break

    policy_history.append(policy.copy())

# =============================================================================
# VISUALIZATION & FILE EXPORT
# =============================================================================
print("\nGenerating and exporting figures...")

# --- Replicating the Policy Progression Map Sequence and Final Value Function ---
fig = plt.figure(figsize=(15, 10))

# Display each policy stage from initialization to optimality
for idx, p_map in enumerate(policy_history):
    ax = fig.add_subplot(2, 3, idx + 1)
    ax.set_title(f"$\\pi_{idx}$", fontsize=14)
    ax.set_xlabel("#Cars at second location")
    ax.set_ylabel("#Cars at first location")
    ax.set_xlim(0, MAX_CARS)
    ax.set_ylim(0, MAX_CARS)

    # Draw contour lines to highlight decision thresholds.
    if np.all(p_map == p_map[0, 0]):
        ax.text(MAX_CARS / 2, MAX_CARS / 2, f"{p_map[0, 0]}", ha='center', va='center', fontsize=14)
    else:
        contours = ax.contour(
            p_map,
            colors='black',
            linewidths=0.5,
            levels=np.arange(-5, 6),
            extent=[0, MAX_CARS, 0, MAX_CARS],
        )
        ax.clabel(contours, inline=True, fontsize=8, fmt='%d')

ax_3d = fig.add_subplot(2, 3, 6, projection='3d')

X = np.arange(num_states)
Y = np.arange(num_states)
X, Y = np.meshgrid(X, Y) # X represents location 2, Y represents location 1

# Plot state-value function surface mesh
ax_3d.plot_wireframe(X, Y, V, color='black', linewidth=0.6)

# Explicitly match view perspectives and label strings to match the textbook's visual format
final_policy_idx = len(policy_history) - 1
ax_3d.set_title(f"$v_{{\\pi_{final_policy_idx}}}$", fontsize=14, pad=20)
ax_3d.set_xlabel('#Cars at second location', labelpad=10)
ax_3d.set_ylabel('#Cars at first location', labelpad=10)
ax_3d.set_zlabel('Value ($)', labelpad=10)

# Adjust viewing angle for visual consistency with Figure 4.2
ax_3d.view_init(elev=30, azim=-125)

plt.suptitle("Figure 4.2: Jack's Car Rental", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig("jacks_car_rental_figure_4_2.png", dpi=300, bbox_inches='tight')
print("-> Saved Figure 4.2 replication as 'jacks_car_rental_figure_4_2.png'")
print("\nExecution completely finished.")
