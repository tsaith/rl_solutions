import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson

# ==========================================
# 控制參數：透過此開關切換原本問題與新問題
# ==========================================
IS_MODIFIED_PROBLEM = True  # True: 執行 Ex 4.7 新問題 ; False: 執行原本的傑克租車問題

# --- 基礎超參數與問題設定 ---
MAX_CARS = 20           # 每個地點最多容納 20 輛車
MAX_MOVE_CARS = 5       # 每晚最多搬運 5 輛車

# 泊松分佈期望值 (λ)
RENTAL_REQUEST_FIRST = 3
RENTAL_REQUEST_SECOND = 4
RETURNS_FIRST = 3
RETURNS_SECOND = 2

# 獎勵與成本設定
RENTAL_CREDIT = 10
MOVE_CAR_COST = 2
ADDITIONAL_PARKING_COST = 4

# 折扣因子與收斂閾值
GAMMA = 0.9
POISSON_UPPER_BOUND = 11  # 泊松分佈截斷點，加速計算用

# 初始化價值函數與策略
V = np.zeros((MAX_CARS + 1, MAX_CARS + 1))
policy = np.zeros((MAX_CARS + 1, MAX_CARS + 1), dtype=int)

# 預先計算 Poisson 機率矩陣以提升效能
poisson_cache = {}
def get_poisson_prob(n, lam):
    key = (n, lam)
    if key not in poisson_cache:
        poisson_cache[key] = poisson.pmf(n, lam)
    return poisson_cache[key]

def expected_return(state, action, state_value, is_modified):
    """
    計算給定狀態與動作下的期望回報 (Expected Return)
    is_modified: 若為 True，則套用 Ex 4.7 的非線性新規則；若為 False，則套用原版規則。
    """
    returns = 0.0
    
    # 1. 計算即時搬運成本 (Action 限制)
    if is_modified:
        # 【新問題】：第一地點搬到第二地點 (action > 0)，第一輛免費
        if action > 0:
            move_cost = (action - 1) * MOVE_CAR_COST
        else:
            move_cost = abs(action) * MOVE_CAR_COST
    else:
        # 【原本問題】：只要搬車，每輛一律收 2 元
        move_cost = abs(action) * MOVE_CAR_COST
        
    returns -= move_cost
    
    # 2. 移車後的夜間車輛數（受限於總容量 20）
    cars_first = int(min(state[0] - action, MAX_CARS))
    cars_second = int(min(state[1] + action, MAX_CARS))
    
    # 3. 超過 10 輛車過夜的額外停車場成本
    if is_modified:
        # 【新問題】：超過 10 輛車過夜要額外收 4 元處罰
        if cars_first > 10:
            returns -= ADDITIONAL_PARKING_COST
        if cars_second > 10:
            returns -= ADDITIONAL_PARKING_COST
    # 【原本問題】：不論過夜幾輛車（不超過 20 輛），皆無額外停車成本

    # 4. 遍歷隔天的租借與歸還組合 (計算期望值)
    for rent_1 in range(POISSON_UPPER_BOUND):
        p_rent_1 = get_poisson_prob(rent_1, RENTAL_REQUEST_FIRST)
        for rent_2 in range(POISSON_UPPER_BOUND):
            p_rent_2 = get_poisson_prob(rent_2, RENTAL_REQUEST_SECOND)
            
            # 實際租出去的車輛數 (不能超過店內現有車輛)
            actual_rent_1 = min(cars_first, rent_1)
            actual_rent_2 = min(cars_second, rent_2)
            
            # 獲得租金獎勵
            reward = (actual_rent_1 + actual_rent_2) * RENTAL_CREDIT
            
            # 租出後的剩餘車輛
            cars_first_remain = cars_first - actual_rent_1
            cars_second_remain = cars_second - actual_rent_2
            
            for return_1 in range(POISSON_UPPER_BOUND):
                p_return_1 = get_poisson_prob(return_1, RETURNS_FIRST)
                for return_2 in range(POISSON_UPPER_BOUND):
                    p_return_2 = get_poisson_prob(return_2, RETURNS_SECOND)
                    
                    # 營業結束後，加上歸還的車輛，並受限於總容量 20
                    next_cars_first = min(cars_first_remain + return_1, MAX_CARS)
                    next_cars_second = min(cars_second_remain + return_2, MAX_CARS)
                    
                    # 聯合機率
                    prob = p_rent_1 * p_rent_2 * p_return_1 * p_return_2
                    
                    # 貝爾曼預期更新
                    returns += prob * (reward + GAMMA * state_value[next_cars_first, next_cars_second])
                    
    return returns

# --- 政策迭代演算法核心 ---

def policy_iteration(is_modified):
    global V, policy
    V.fill(0)       # 重置價值函數
    policy.fill(0)  # 重置策略
    iterations = 0
    
    problem_type = "Ex 4.7 新問題" if is_modified else "原本問題"
    print(f"================ 啟動：{problem_type} 政策迭代 ================")
    
    while True:
        print(f"\n--- 進入第 {iterations} 次策略迭代 ---")
        
        # 步驟 1: 策略評估 (Policy Evaluation)
        theta = 0.1
        while True:
            delta = 0.0
            V_old = V.copy()  # 用於同步更新
            for i in range(MAX_CARS + 1):
                for j in range(MAX_CARS + 1):
                    old_v = V[i, j]
                    V[i, j] = expected_return((i, j), policy[i, j], V_old, is_modified)
                    delta = max(delta, abs(old_v - V[i, j]))
            print(f"策略評估中... Delta: {delta:.4f}")
            if delta < theta:
                break
        
        # 步驟 2: 策略提升 (Policy Improvement)
        policy_stable = True
        for i in range(MAX_CARS + 1):
            for j in range(MAX_CARS + 1):
                old_action = policy[i, j]
                
                # 計算所有合法搬運動作的範圍
                min_act = -min(j, MAX_MOVE_CARS)
                max_act = min(i, MAX_MOVE_CARS)
                legal_actions = np.arange(min_act, max_act + 1)
                
                # 預防 Ex 4.4 死循環 Bug 修正
                q_old = expected_return((i, j), old_action, V, is_modified)
                best_action = old_action
                best_q = q_old
                
                for act in legal_actions:
                    q_val = expected_return((i, j), act, V, is_modified)
                    # 僅在「嚴格優於」當前動作價值時，才更新動作 (Ex 4.4 思想)
                    if q_val > best_q + 1e-5: 
                        best_q = q_val
                        best_action = act
                
                policy[i, j] = best_action
                if old_action != policy[i, j]:
                    policy_stable = False
                    
        iterations += 1
        if policy_stable:
            print(f"策略已成功收斂！總計迭代次數: {iterations}")
            break

# 執行指定的設定
policy_iteration(is_modified=IS_MODIFIED_PROBLEM)

# --- 成果繪圖呈現 ---
plt.figure(figsize=(10, 8))
# 使用二維矩陣熱圖展示各個 (地點1車輛數, 地點2車輛數) 狀態下的最佳搬運車數
plt.imshow(policy, cmap='bwr', origin='lower', extent=[0, MAX_CARS, 0, MAX_CARS], vmin=-5, vmax=5)
cbar = plt.colorbar()
cbar.set_ticks(np.arange(-5, 6))
cbar.set_label('Action: Move from First to Second (Negative means Second to First)', fontsize=11)

title_suffix = "Exercise 4.7 (Modified Problem)" if IS_MODIFIED_PROBLEM else "Original Problem"
plt.title(f"Jack's Car Rental - Optimal Policy\n[{title_suffix}]", fontsize=14, fontweight='bold')
plt.xlabel("Number of Cars at Second Location", fontsize=12)
plt.ylabel("Number of Cars at First Location", fontsize=12)

# 畫上 10 輛車的分界輔助線，方便觀察 Ex 4.7 的非線性邊界
if IS_MODIFIED_PROBLEM:
    plt.axhline(10, color='black', linestyle=':', alpha=0.6)
    plt.axvline(10, color='black', linestyle=':', alpha=0.6)

plt.grid(True, which='both', color='gray', linestyle='--', linewidth=0.3, alpha=0.5)
plt.show()