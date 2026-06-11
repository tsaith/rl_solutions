import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

# ==========================================
# 1. 環境與基本參數設定
# ==========================================
num_arms = 10
total_steps = 2000 #200000
measure_steps = 1000  #100000
num_runs = 2000 #2000  # 書中通常使用 2000 次獨立實驗來取得穩定的平均值

# 參數軸 (2的冪次：1/128, 1/64, ..., 4)
param_pow = np.arange(-7, 3, dtype=float)
params = 2 ** param_pow

# 初始化用於繪圖的結果字典
results = {
    'epsilon-greedy (constant alpha=0.1)': [],
    'sample-average epsilon-greedy': [],
    'optimistic initialization (constant alpha=0.1)': [],
    'UCB': [],
    'gradient bandit': []
}

# Figure output
output_path = Path(__file__).with_name("2_11.png")

# ==========================================
# 2. 演算法核心邏輯定義
# ==========================================

def run_bandit(algo_name, param_val):
    """
    模擬非平穩強盜機環境並計算單一參數設定下的長遠平均回報。
    """
    rewards_sum = 0.0
    
    # 為了加速運算，我們可以平行或逐次跑不同的獨立隨機任務
    for run in range(num_runs):
        # 初始化環境：非平穩環境，一開始所有臂的 q_star 均為 0
        q_star = np.zeros(num_arms)
        
        # 初始化智能體記憶
        Q = np.zeros(num_arms)
        N = np.zeros(num_arms)
        
        # 梯度強盜機專用偏好與基線
        H = np.zeros(num_arms)
        running_avg_reward = 0.0
        
        # 樂觀初始值專用
        if algo_name == 'optimistic initialization (constant alpha=0.1)':
            Q.fill(param_val)  # 參數值此時代表初始的 Q1
            
        run_rewards = 0.0
        
        for step in range(1, total_steps + 1):
            # --- 動作選擇階段 ---
            if algo_name in ['epsilon-greedy (constant alpha=0.1)', 'sample-average epsilon-greedy']:
                if np.random.rand() < param_val: # 此時參數代表 epsilon
                    action = np.random.randint(num_arms)
                else:
                    action = np.argmax(Q)
                    
            elif algo_name == 'optimistic initialization (constant alpha=0.1)':
                # 樂觀初始值法在 Figure 2.6 中使用 greedy action selection (epsilon = 0)
                action = np.argmax(Q)
                
            elif algo_name == 'UCB':
                if step <= num_arms:
                    action = step - 1
                else:
                    # 避免分母為 0
                    ucb_values = Q + param_val * np.sqrt(np.log(step) / N) # 參數代表 c
                    action = np.argmax(ucb_values)
                    
            elif algo_name == 'gradient bandit':
                # Softmax 分布計算策略
                exp_H = np.exp(H - np.max(H)) # 減去最大值防止溢位
                probs = exp_H / np.sum(exp_H)
                action = np.random.choice(num_arms, p=probs)

            # --- 環境與回報反饋 ---
            # 真實獎勵會加上一個標準差為 1 的隨機高斯雜訊
            reward = np.random.normal(q_star[action], 1.0)
            
            # 關鍵：非平穩環境演進 (Exercise 2.5)
            # 每一步所有臂的 q_star 都會加上一個獨立的高斯隨機走路雜訊
            q_star += np.random.normal(0.0, 0.01, num_arms)
            
            # --- 收集最後 100,000 步的表現 ---
            if step > (total_steps - measure_steps):
                run_rewards += reward

            # --- 智能體學習更新階段 ---
            N[action] += 1
            
            if algo_name == 'epsilon-greedy (constant alpha=0.1)' or algo_name == 'optimistic initialization (constant alpha=0.1)':
                Q[action] += 0.1 * (reward - Q[action])
                
            elif algo_name == 'sample-average epsilon-greedy':
                Q[action] += (1.0 / N[action]) * (reward - Q[action])
                
            elif algo_name == 'UCB':
                Q[action] += 0.1 * (reward - Q[action]) # UCB在非平穩環境中也通常搭配固定步長
                
            elif algo_name == 'gradient bandit':
                # 參數值此時代表學習率 alpha
                alpha = param_val
                running_avg_reward += (1.0 / step) * (reward - running_avg_reward)
                
                # 更新策略偏好
                for a in range(num_arms):
                    if a == action:
                        H[a] += alpha * (reward - running_avg_reward) * (1.0 - probs[a])
                    else:
                        H[a] -= alpha * (reward - running_avg_reward) * probs[a]
                        
        rewards_sum += (run_rewards / measure_steps)
        
    return rewards_sum / num_runs

# ==========================================
# 3. 執行模擬（參數掃描）
# ==========================================
print("開始跑非平穩環境下的多演算法參數掃描...")

for algo in results.keys():
    print(f"正在評估演算法: {algo} ...")
    for p in tqdm(params):
        avg_rew = run_bandit(algo, p)
        results[algo].append(avg_rew)

# ==========================================
# 4. 數據可視化繪圖
# ==========================================
plt.figure(figsize=(10, 8))

# 定義每條曲線在原書中的對應樣式與顏色
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

# 設定與書中一致的圖表樣式
plt.xlabel(r'Parameter ($2^x$: $\epsilon$, $\alpha$, $c$, $Q_0$)', fontsize=12)
plt.ylabel('Average Reward over last 100,000 steps', fontsize=12)
plt.title('Exercise 2.11: Parameter Study in Nonstationary Environments', fontsize=14)
plt.xticks(param_pow, ['1/128', '1/64', '1/32', '1/16', '1/8', '1/4', '1/2', '1', '2', '4'])
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=2)
plt.tight_layout()
plt.savefig(output_path, dpi=150)
plt.show()
