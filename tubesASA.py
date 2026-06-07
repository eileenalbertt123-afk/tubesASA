# nama: eileen Albert Tandrio
# nim: 24060124140180

import random
import math
import time
import numpy as np

# Reproducibility 
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Parameters 
LAMBDA      = 10       # fairness weight
NUM_REGIONS = 4        # jumlah wilayah
SA_RUNS     = 5        # jumlah run SA per skenario

SA_T0       = 1000
SA_T_MIN    = 0.01
SA_ALPHA    = 0.995
SA_ITER_PER_TEMP = 100

# Dataset Generator 
def generate_dataset(n, num_regions=NUM_REGIONS, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    costs    = [random.randint(10, 100) for _ in range(n)]   # juta rupiah
    urgency  = [random.randint(1, 10)   for _ in range(n)]
    regions  = [random.randint(0, num_regions - 1) for _ in range(n)]
    budget   = int(0.4 * sum(costs))
    return costs, urgency, regions, budget

# Jain's Fairness Index 
def jains_fairness(x, regions, num_regions=NUM_REGIONS):
    counts = [0] * num_regions
    for i, xi in enumerate(x):
        if xi:
            counts[regions[i]] += 1
    total = sum(counts)
    if total == 0:
        return 0.0
    num   = sum(counts) ** 2
    denom = num_regions * sum(c ** 2 for c in counts)
    return num / denom if denom > 0 else 0.0

# Objective Function 
def objective(x, costs, urgency, regions, budget, theta=0.5):
    total_cost = sum(costs[i] for i in range(len(x)) if x[i])
    if total_cost > budget:
        return -1e9   # penalti pelanggaran kendala anggaran
    jfi = jains_fairness(x, regions)
    if jfi < theta:
        return -1e9   # penalti pelanggaran kendala fairness minimum
    total_urgency = sum(urgency[i] for i in range(len(x)) if x[i])
    return total_urgency + LAMBDA * jfi

# Brute Force 
def brute_force(costs, urgency, regions, budget):
    n = len(costs)
    best_val = -1e18
    best_x   = [0] * n
    for mask in range(1 << n):
        x = [(mask >> i) & 1 for i in range(n)]
        val = objective(x, costs, urgency, regions, budget)
        if val > best_val:
            best_val = val
            best_x   = x[:]
    best_val = objective(best_x, costs, urgency, regions, budget)
    jfi = jains_fairness(best_x, regions)
    return best_x, best_val, jfi

# Dynamic Programming 
def dynamic_programming(costs, urgency, regions, budget):
    n = len(costs)
    dp = [[0] * (budget + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        w = costs[i - 1]
        v = urgency[i - 1]
        for b in range(budget + 1):
            dp[i][b] = dp[i - 1][b]
            if w <= b and dp[i - 1][b - w] + v > dp[i][b]:
                dp[i][b] = dp[i - 1][b - w] + v

    # Traceback
    x = [0] * n
    b = budget
    for i in range(n, 0, -1):
        if dp[i][b] != dp[i - 1][b]:
            x[i - 1] = 1
            b -= costs[i - 1]

    jfi = jains_fairness(x, regions)
    val = sum(urgency[i] for i in range(n) if x[i]) + LAMBDA * jfi
    return x, val, jfi

# Simulated Annealing 
def simulated_annealing_single(costs, urgency, regions, budget):
    n = len(costs)
    # Solusi awal acak yang feasible
    x = [0] * n
    indices = list(range(n))
    random.shuffle(indices)
    cap = 0
    for i in indices:
        if cap + costs[i] <= budget:
            x[i] = 1
            cap += costs[i]

    cur_val = objective(x, costs, urgency, regions, budget)
    best_x  = x[:]
    best_val = cur_val

    T = SA_T0
    while T > SA_T_MIN:
        for _ in range(SA_ITER_PER_TEMP):
            # Neighborhood move: flip satu bit
            j = random.randint(0, n - 1)
            x_new = x[:]
            x_new[j] = 1 - x_new[j]
            new_val = objective(x_new, costs, urgency, regions, budget)
            delta = new_val - cur_val
            if delta > 0 or random.random() < math.exp(delta / T):
                x   = x_new
                cur_val = new_val
                if cur_val > best_val:
                    best_val = cur_val
                    best_x   = x[:]
        T *= SA_ALPHA

    jfi = jains_fairness(best_x, regions)
    return best_x, best_val, jfi

def simulated_annealing(costs, urgency, regions, budget, runs=SA_RUNS):
    results = [simulated_annealing_single(costs, urgency, regions, budget)
               for _ in range(runs)]
    best = max(results, key=lambda r: r[1])
    avg_val = sum(r[1] for r in results) / runs
    avg_jfi = sum(r[2] for r in results) / runs
    return best[0], best[1], best[2], avg_val, avg_jfi

# Runner 
def run_experiment(n, dataset_seed=SEED):
    costs, urgency, regions, budget = generate_dataset(n, seed=dataset_seed)
    results = {}

    # Brute Force (hanya untuk n <= 20)
    if n <= 20:
        t0 = time.perf_counter()
        bf_x, bf_val, bf_jfi = brute_force(costs, urgency, regions, budget)
        bf_time = (time.perf_counter() - t0) * 1000
        results['BF'] = {'val': bf_val, 'jfi': bf_jfi, 'time_ms': bf_time, 'x': bf_x}
    else:
        results['BF'] = None

    # Dynamic Programming
    t0 = time.perf_counter()
    dp_x, dp_val, dp_jfi = dynamic_programming(costs, urgency, regions, budget)
    dp_time = (time.perf_counter() - t0) * 1000
    results['DP'] = {'val': dp_val, 'jfi': dp_jfi, 'time_ms': dp_time, 'x': dp_x}

    # Simulated Annealing
    t0 = time.perf_counter()
    sa_x, sa_best_val, sa_best_jfi, sa_avg_val, sa_avg_jfi = \
        simulated_annealing(costs, urgency, regions, budget)
    sa_time = (time.perf_counter() - t0) * 1000
    results['SA'] = {
        'val': sa_best_val, 'jfi': sa_best_jfi,
        'avg_val': sa_avg_val, 'avg_jfi': sa_avg_jfi,
        'time_ms': sa_time, 'x': sa_x
    }

    return results, costs, urgency, regions, budget

# Main
if __name__ == '__main__':
    sizes = [10, 15, 20, 50]
    print(f"{'n':>4} | {'Algo':>4} | {'Obj Val':>10} | {'JFI':>6} | {'Time (ms)':>10} | {'Gap vs BF':>10}")
    print("-" * 65)

    for n in sizes:
        results, costs, urgency, regions, budget = run_experiment(n)
        bf_val = results['BF']['val'] if results['BF'] else None

        for algo in ['BF', 'DP', 'SA']:
            r = results[algo]
            if r is None:
                print(f"{n:>4} | {algo:>4} | {'N/A (n>20)':>10} | {'–':>6} | {'–':>10} | {'–':>10}")
                continue
            val  = r['val']
            jfi  = r['jfi']
            t_ms = r['time_ms']
            gap  = f"{((bf_val - val) / bf_val * 100):.2f}%" if bf_val and algo != 'BF' else "–"
            print(f"{n:>4} | {algo:>4} | {val:>10.4f} | {jfi:>6.4f} | {t_ms:>10.4f} | {gap:>10}")
            if algo == 'SA':
                print(f"    avg_val={r['avg_val']:.4f} | avg_jfi={r['avg_jfi']:.4f}")
        print()
