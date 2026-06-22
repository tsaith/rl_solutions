import numpy as np
import matplotlib.pyplot as plt

def solve_gambler(p_h, theta=1e-9):
    """
    Solves the Gambler's Problem using Value Iteration.
    
    Parameters:
        p_h (float): Probability of the coin coming up heads.
        theta (float): Convergence threshold. Default is a very small number to explore stability.
    """
    # Define states: 0 and 100 are dummy terminal states
    GOAL = 100
    states = np.arange(GOAL + 1)
    
    # Initialize value function
    # Terminal state 100 has value 1.0, all other states initialized to 0.0
    V = np.zeros(GOAL + 1)
    V[GOAL] = 1.0
    
    # Track value updates over sweeps for visualization
    sweeps_history = []
    sweep_count = 0
    
    # --- Value Iteration Loop ---
    while True:
        delta = 0.0
        # Copy values to perform synchronous or stable updates
        V_old = V.copy()
        
        # Update only non-terminal states (1 to 99)
        for s in range(1, GOAL):
            actions = np.arange(0, min(s, GOAL - s) + 1)
            action_values = []
            
            for a in actions:
                # Bellman Optimality Equation:
                # Expected value = p_h * V(s + a) + (1 - p_h) * V(s - a)
                val = p_h * V_old[s + a] + (1.0 - p_h) * V_old[s - a]
                action_values.append(val)
                
            best_value = np.max(action_values)
            delta = max(delta, abs(V_old[s] - best_value))
            V[s] = best_value
            
        sweep_count += 1
        # Capture early sweeps and late-stage convergence profiles
        if sweep_count in [1, 2, 3, 5, 10, 50] or delta < theta:
            sweeps_history.append((sweep_count, V.copy()))
            
        if delta < theta:
            print(f"p_h = {p_h}: Value Iteration converged after {sweep_count} sweeps.")
            break

    # --- Extract Final Optimal Policy ---
    policy = np.zeros(GOAL + 1, dtype=int)
    for s in range(1, GOAL):
        actions = np.arange(1, min(s, GOAL - s) + 1) # Action 0 is excluded for active betting
        action_values = []
        
        for a in actions:
            val = p_h * V[s + a] + (1.0 - p_h) * V[s - a]
            action_values.append(val)
            
        # Due to the multi-tie phenomenon in argmax, rounding prevents floating point noise
        # from selecting arbitrary actions when values are identical.
        max_val = np.max(action_values)
        best_actions = actions[np.isclose(action_values, max_val, atol=1e-8)]
        
        # Replicating the textbook choice: picking the smallest optimal stake or the first peak
        policy[s] = best_actions[0]

    return sweeps_history, V, policy

# =============================================================================
# VISUALIZATION & EXPORT
# =============================================================================
def plot_results(p_h, sweeps_history, policy):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # 1. Plot Value Function Sweeps
    for sweep, v_profile in sweeps_history:
        if sweep in [1, 2, 3]:
            ax1.plot(v_profile[1:100], label=f'Sweep {sweep}')
        elif sweep == sweeps_history[-1][0]:
            ax1.plot(v_profile[1:100], label=f'Final Value Function (Sweep {sweep})', color='black', linewidth=2)
        else:
            ax1.plot(v_profile[1:100], alpha=0.5, linestyle='--')
            
    ax1.set_title(f"Value Function Estimates ($p_h = {p_h}$)", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Capital")
    ax1.set_ylabel("Value Estimates (Winning Probability)")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)
    
    # 2. Plot Final Policy
    ax2.step(np.arange(1, 100), policy[1:100], where='mid', color='blue', linewidth=1.5)
    ax2.set_title(f"Final Policy (Stakes) ($p_h = {p_h}$)", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Capital")
    ax2.set_ylabel("Final Policy (Stake)")
    ax2.set_ylim(0, 51)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = f"gambler_problem_ph_{int(p_h*100)}.png"
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"-> Successfully saved visual results as '{filename}'")

# Execute for both problem configurations
sweeps_25, V_25, policy_25 = solve_gambler(p_h=0.25)
plot_results(0.25, sweeps_25, policy_25)

sweeps_55, V_55, policy_55 = solve_gambler(p_h=0.55)
plot_results(0.55, sweeps_55, policy_55)