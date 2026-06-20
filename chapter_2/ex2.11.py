import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

# ==========================================
# 1. Environment and basic parameter settings
# ==========================================
num_arms = 10
total_steps = 20000 #200000
measure_steps = 10000  #100000
num_runs = 2000 #2000  # The book usually uses 2,000 independent runs for stable averages.
random_seed = 0

np.random.seed(random_seed)

# Parameter axis (powers of 2: 1/128, 1/64, ..., 4)
param_pow = np.arange(-7, 3, dtype=float)
params = 2 ** param_pow

# Initialize the results dictionary for plotting.
results = {
    'epsilon-greedy (constant alpha=0.1)': [],
    'sample-average epsilon-greedy': [],
    'optimistic initialization (constant alpha=0.1)': [],
    'UCB': [],
    'gradient bandit': []
}

# Figure output
output_path = Path(__file__).with_name("ex2.11.png")

# ==========================================
# 2. Core algorithm logic
# ==========================================

def run_bandit(algo_name, param_val):
    """
    Simulate the nonstationary bandit environment and return the long-run average reward.
    """
    rewards_sum = 0.0
    
    # Run independent random tasks sequentially. These could be parallelized for speed.
    for run in range(num_runs):
        # Initialize the environment: all q_star values start at 0 in the nonstationary case.
        q_star = np.zeros(num_arms)
        
        # Initialize agent memory.
        Q = np.zeros(num_arms)
        N = np.zeros(num_arms)
        
        # Preferences and baseline for the gradient bandit algorithm.
        H = np.zeros(num_arms)
        running_avg_reward = 0.0
        
        # Optimistic initialization only.
        if algo_name == 'optimistic initialization (constant alpha=0.1)':
            Q.fill(param_val)  # The parameter value represents the initial Q1.
            
        run_rewards = 0.0
        
        for step in range(1, total_steps + 1):
            # --- Action selection ---
            if algo_name in ['epsilon-greedy (constant alpha=0.1)', 'sample-average epsilon-greedy']:
                if np.random.rand() < param_val: # The parameter represents epsilon.
                    action = np.random.randint(num_arms)
                else:
                    action = np.argmax(Q)
                    
            elif algo_name == 'optimistic initialization (constant alpha=0.1)':
                # Optimistic initialization uses greedy action selection in Figure 2.6 (epsilon = 0).
                action = np.argmax(Q)
                
            elif algo_name == 'UCB':
                if step <= num_arms:
                    action = step - 1
                else:
                    # Avoid division by zero.
                    ucb_values = Q + param_val * np.sqrt(np.log(step) / N) # The parameter represents c.
                    action = np.argmax(ucb_values)
                    
            elif algo_name == 'gradient bandit':
                # Compute the policy with a softmax distribution.
                # Adding or subtracting the same constant from numerator and denominator leaves probabilities unchanged.
                exp_H = np.exp(H - np.max(H)) # Subtract the maximum value to prevent overflow.
                probs = exp_H / np.sum(exp_H)
                action = np.random.choice(num_arms, p=probs)

            # --- Environment and reward feedback ---
            # The true reward is perturbed by Gaussian noise with standard deviation 1.
            reward = np.random.normal(q_star[action], 1.0)
            
            # Key nonstationary environment evolution from Exercise 2.5.
            # At each step, every arm's q_star receives an independent Gaussian random-walk increment.
            q_star += np.random.normal(0.0, 0.01, num_arms)
            
            # --- Collect performance over the last 100,000 steps ---
            if step > (total_steps - measure_steps):
                run_rewards += reward

            # --- Agent learning update ---
            N[action] += 1
            
            if algo_name == 'epsilon-greedy (constant alpha=0.1)' or algo_name == 'optimistic initialization (constant alpha=0.1)':
                Q[action] += 0.1 * (reward - Q[action])
                
            elif algo_name == 'sample-average epsilon-greedy':
                Q[action] += (1.0 / N[action]) * (reward - Q[action])
                
            elif algo_name == 'UCB':
                Q[action] += 0.1 * (reward - Q[action]) # UCB is also commonly paired with a constant step size in nonstationary environments.
                
            elif algo_name == 'gradient bandit':
                # The parameter value represents the learning rate alpha.
                alpha = param_val
                running_avg_reward += (1.0 / step) * (reward - running_avg_reward)
                
                # Update action preferences.
                for a in range(num_arms):
                    if a == action:
                        H[a] += alpha * (reward - running_avg_reward) * (1.0 - probs[a])
                    else:
                        H[a] -= alpha * (reward - running_avg_reward) * probs[a]
                        
        rewards_sum += (run_rewards / measure_steps)
        
    return rewards_sum / num_runs

# ==========================================
# 3. Run the simulation (parameter sweep)
# ==========================================
print("Starting the multi-algorithm parameter sweep in the nonstationary environment...")

for algo in results.keys():
    print(f"Evaluating algorithm: {algo} ...")
    for p in tqdm(params):
        avg_rew = run_bandit(algo, p)
        results[algo].append(avg_rew)

# ==========================================
# 4. Data visualization
# ==========================================
plt.figure(figsize=(10, 8))

# Define each curve's style and color to match the book.
styles = {
    'epsilon-greedy (constant alpha=0.1)': ('red', '-'),
    'sample-average epsilon-greedy': ('red', '--'),
    'optimistic initialization (constant alpha=0.1)': ('blue', '-'),
    'UCB': ('blue', '--'),
    'gradient bandit': ('green', '-')
}

for algo, data in results.items():
    color, linestyle = styles[algo]
    plt.plot(param_pow, data, label=algo, color=color, linestyle=linestyle, marker='o')

# Configure plot styling to match the book.
plt.xlabel(r'Parameter ($2^x$: $\epsilon$, $\alpha$, $c$, $Q_0$)', fontsize=12)
plt.ylabel(f'Average Reward over last {measure_steps} steps', fontsize=12)
plt.title('Exercise 2.11: Parameter Study in Nonstationary Environments', fontsize=14)
plt.xticks(param_pow, ['1/128', '1/64', '1/32', '1/16', '1/8', '1/4', '1/2', '1', '2', '4'])
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=2)
plt.tight_layout()
plt.savefig(output_path, dpi=150)
plt.show()
