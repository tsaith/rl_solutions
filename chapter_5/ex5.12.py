import numpy as np
import matplotlib.pyplot as plt
import random

# =============================================================================
# ENVIRONMENT & TRACK SETUP (Exercise 5.12)
# =============================================================================
rows, cols = 32, 17
epsilon = 0.1
gamma = 1.0

# Actions: acceleration (ax, ay) in {-1, 0, 1}
ACTIONS = [
    (0, 0),
    (0, 1),
    (0, -1),
    (1, 0),
    (1, 1),
    (1, -1),
    (-1, 0),
    (-1, 1),
    (-1, -1),
]
num_actions = len(ACTIONS)

# Velocity components are restricted to be non-negative and less than 5,
# and they cannot be both zero.
vel_len = 5

# Set up the 1st race track map.
# 0 represents track (safe), 1 represents boundary/out-of-bounds.
# In Python, row 0 is the bottom-most row (start line), row 31 is the top-most row.
track = np.zeros((rows, cols), dtype=np.int8)

# Boundary setup to match the Julia implementation:
track[31, 0:3] = 1
track[30, 0:2] = 1
track[29, 0:2] = 1
track[28, 0] = 1
track[0:18, 0] = 1
track[0:10, 1] = 1
track[0:3, 2] = 1
track[0:26, 9:17] = 1
track[25, 9] = 0  # ensure this cell is open

# Start line: columns 3 to 8 on row 0.
start_cols = np.arange(3, 9)

# Finish line cells: row 26 to 31, column 16.
fin_cells = {(r, 16) for r in range(26, 32)}

# Precompute valid actions for each velocity combination (h, v) where h, v in 0..4.
# h is horizontal velocity, v is vertical velocity.
valid_actions = {}
for h in range(vel_len):
    for v in range(vel_len):
        acts = []
        for a_idx, (ax, ay) in enumerate(ACTIONS):
            next_h = h + ax
            next_v = v + ay
            if (0 <= next_h < vel_len) and (0 <= next_v < vel_len):
                if not (next_h == 0 and next_v == 0):
                    acts.append(a_idx)
        valid_actions[(h, v)] = acts

# =============================================================================
# INITIALIZE Q, C, and PI (Off-policy MC Control)
# =============================================================================
# Q(s, a): state is (row, col, vel_h, vel_v), action is a_idx.
# Initialized optimistically to encourage exploration.
Q = np.random.rand(rows, cols, vel_len, vel_len, num_actions) * 400.0 - 500.0
C = np.zeros((rows, cols, vel_len, vel_len, num_actions), dtype=np.float64)

# Initialize policy pi greedily based on Q.
policy = np.zeros((rows, cols, vel_len, vel_len), dtype=np.int32)
for r in range(rows):
    for c in range(cols):
        for h in range(vel_len):
            for v in range(vel_len):
                acts = valid_actions[(h, v)]
                best_act = acts[np.argmax(Q[r, c, h, v, acts])]
                policy[r, c, h, v] = best_act

# =============================================================================
# TRAJECTORY GENERATION
# =============================================================================
def make_trajectory(eps, noise=True):
    S = []
    A = []
    B = []

    # Choose a random starting column on row 0 with velocity (0, 0)
    start_c = random.choice(start_cols)
    s = (0, start_c, 0, 0)
    S.append(s)

    while True:
        r, c, h, v = s
        acts = valid_actions[(h, v)]
        num_acts = len(acts)

        pia = policy[r, c, h, v]
        pia_valid = pia in acts

        # Select action using epsilon-greedy policy
        if random.random() >= eps:
            if pia_valid:
                a = pia
                b = 1.0 - eps + eps / num_acts
            else:
                a = random.choice(acts)
                b = 1.0 / num_acts
        else:
            a = random.choice(acts)
            b = (eps if pia_valid else 1.0) / num_acts

        # Add velocity noise: 10% chance that acceleration becomes (0, 0)
        if noise and random.random() < 0.1:
            a = 0  # ACTIONS[0] is (0, 0)
            b = 0.1

        A.append(a)
        B.append(b)

        ax, ay = ACTIONS[a]
        next_h = h + ax
        next_v = v + ay

        next_r = r + next_v
        next_c = c + next_h

        # Check if the path intersects the finish line or hits a boundary.
        max_vel = max(next_h, next_v)
        crossed = False
        collided = False
        for i in range(1, max_vel + 1):
            path_r = min(r + i, next_r)
            path_c = min(c + i, next_c)
            if (path_r, path_c) in fin_cells:
                crossed = True
                break
            if (not (0 <= path_r < rows and 0 <= path_c < cols)) or (track[path_r, path_c] == 1):
                collided = True
                break

        if crossed:
            break

        if collided:
            # Collision: reset to start line with zero velocity
            next_start_c = random.choice(start_cols)
            s = (0, next_start_c, 0, 0)
        else:
            s = (next_r, next_c, next_h, next_v)

        S.append(s)

    return S, A, B

# =============================================================================
# OFF-POLICY MONTE CARLO UPDATE
# =============================================================================
def run_episode(S, A, B):
    G = 0.0
    W = 1.0
    R = -1.0
    T = len(A)

    for t in range(T - 1, -1, -1):
        r, c, h, v = S[t]
        a = A[t]

        G = gamma * G + R
        C[r, c, h, v, a] += W
        Q[r, c, h, v, a] += W * (G - Q[r, c, h, v, a]) / C[r, c, h, v, a]

        # Update policy greedily
        acts = valid_actions[(h, v)]
        best_act = acts[np.argmax(Q[r, c, h, v, acts])]
        policy[r, c, h, v] = best_act

        if a != best_act:
            # Policy diverged from target policy, stop backward update
            return t

        W /= B[t]

    return -1

# =============================================================================
# MAIN TRAINING LOOP
# =============================================================================
def train(episodes=100000):
    print(f"Starting Off-policy Monte Carlo training for {episodes} episodes...")
    rewards = []
    
    for i in range(1, episodes + 1):
        # Generate trajectory using behavior policy (epsilon-greedy with noise)
        S, A, B = make_trajectory(epsilon, noise=True)
        # Run backward MC update
        _ = run_episode(S, A, B)
        
        # Periodically track performance (greedy, no noise)
        if i % 1000 == 0:
            S_eval, A_eval, _ = make_trajectory(0.0, noise=False)
            reward_eval = -len(A_eval)
            rewards.append(reward_eval)
            print(f"Episode {i:6d} | Eval Steps: {len(A_eval):4d} | Return: {reward_eval:4d}")

    return rewards

# =============================================================================
# PATH PLOTTING & VISUALIZATION
# =============================================================================
def generate_best_path():
    """
    Tries to generate an optimal path (no noise, epsilon=0) from each of the 
    start columns. Returns the path that successfully reaches the finish line in the 
    fewest steps.
    """
    best_path = None
    min_steps = float('inf')

    for sc in start_cols:
        s = (0, sc, 0, 0)
        path = [s]
        steps = 0
        stuck = False
        
        while True:
            r, c, h, v = s
            acts = valid_actions[(h, v)]
            pia = policy[r, c, h, v]
            ax, ay = ACTIONS[pia]
            next_h = h + ax
            next_v = v + ay

            next_r = r + next_v
            next_c = c + next_h

            max_vel = max(next_h, next_v)
            crossed = False
            collided = False
            for i in range(1, max_vel + 1):
                path_r = min(r + i, next_r)
                path_c = min(c + i, next_c)
                if (path_r, path_c) in fin_cells:
                    crossed = True
                    path.append((path_r, path_c, next_h, next_v))
                    break
                if (not (0 <= path_r < rows and 0 <= path_c < cols)) or (track[path_r, path_c] == 1):
                    collided = True
                    break

            if crossed:
                break

            if collided:
                # Boundary hit in deterministic evaluation indicates sub-optimal policy or stuck loop
                stuck = True
                break

            s = (next_r, next_c, next_h, next_v)
            path.append(s)
            steps += 1
            if steps > 100:
                stuck = True
                break

        if not stuck and len(path) < min_steps:
            min_steps = len(path)
            best_path = path

    return best_path

def plot_track_and_path(best_path):
    fig, ax = plt.subplots(figsize=(8, 12))
    
    # Create track image (RGB)
    # 1.0 = White (safe track)
    # [0.7, 0.7, 0.7] = Gray (out of bounds)
    grid_img = np.ones((rows, cols, 3))
    for r in range(rows):
        for c in range(cols):
            if track[r, c] == 1:
                grid_img[r, c] = [0.7, 0.7, 0.7]

    # Paint start line in green
    for c in start_cols:
        grid_img[0, c] = [0.2, 0.8, 0.2]

    # Paint finish line in red
    for r in range(26, 32):
        grid_img[r, 16] = [0.8, 0.2, 0.2]

    # Plot track using origin='lower' so row 0 is at the bottom
    ax.imshow(grid_img, origin='lower')

    # Draw grid lines to see grid cells clearly
    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=0.5)
    ax.tick_params(which='both', bottom=False, left=False, labelbottom=True, labelleft=True)

    # Plot optimal path if found
    if best_path is not None:
        path_cols = [s[1] for s in best_path]
        path_rows = [s[0] for s in best_path]
        
        # Plot path as line
        ax.plot(path_cols, path_rows, color='blue', linewidth=3, label='Optimal Path')
        # Mark path points (start point is green, end is red, intermediate points are blue)
        ax.scatter(path_cols[0], path_rows[0], color='darkgreen', s=150, zorder=5, label='Start State')
        ax.scatter(path_cols[-1], path_rows[-1], color='darkred', s=150, zorder=5, label='Finish State')
        ax.scatter(path_cols[1:-1], path_rows[1:-1], color='cyan', s=60, zorder=4)

        # Draw direction arrows between path points
        for idx in range(len(best_path) - 1):
            r1, c1, _, _ = best_path[idx]
            r2, c2, _, _ = best_path[idx+1]
            ax.annotate('', xy=(c2, r2), xytext=(c1, r1),
                        arrowprops=dict(arrowstyle="->", color="navy", lw=2, shrinkA=5, shrinkB=5))

    ax.set_title("Racetrack Optimal Path (Exercise 5.12)", fontsize=14, fontweight='bold')
    ax.legend(loc='upper left')
    
    # Save visualization to file
    plt.tight_layout()
    plt.savefig("racetrack_optimal_path.png", dpi=300)
    plt.close()
    print("-> Saved optimal racetrack visualization to 'racetrack_optimal_path.png'")

if __name__ == "__main__":
    # Train policy
    rewards = train(episodes=120000)
    
    # Generate best path and plot it
    best_path = generate_best_path()
    if best_path is not None:
        print(f"Optimal path found with {len(best_path) - 1} steps:")
        for idx, s in enumerate(best_path):
            print(f"  Step {idx:2d}: Position ({s[0]:2d}, {s[1]:2d}) | Velocity ({s[2]:1d}, {s[3]:1d})")
        plot_track_and_path(best_path)
    else:
        print("Could not find a valid complete path without collision after training. Try training for more episodes.")
