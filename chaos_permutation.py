"""
混沌置乱的循环阶分析
Cycle Order Analysis of Chaotic Permutations

实现三种混沌映射（Logistic、Tent、Sine），生成置乱表并分析其循环结构。
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from math import gcd
from functools import reduce
from collections import Counter
import time
import os

# ============================================================
# 1. 混沌映射定义
# ============================================================

def logistic_map(x, mu=3.99):
    """Logistic映射: x_{n+1} = mu * x_n * (1 - x_n), mu in (3.57, 4)"""
    return mu * x * (1 - x)

def tent_map(x, mu=1.99):
    """Tent映射: x_{n+1} = mu * min(x, 1-x), mu in (1, 2)"""
    return mu * min(x, 1 - x)

def sine_map(x, mu=0.99):
    """Sine映射: x_{n+1} = mu * sin(pi * x), mu in (0, 1)"""
    return mu * np.sin(np.pi * x)


# ============================================================
# 2. 基于混沌序列生成置乱表
# ============================================================

def generate_permutation(chaos_func, x0, N, M=1000, **kwargs):
    """
    基于混沌映射生成置乱表。
    1. 选定参数和初始值x0, 迭代M轮得到x_M (丢弃暂态)
    2. 继续迭代计算 x_{M+1} ~ x_{M+N}
    3. 将N个数排序, 以位置为置乱索引
    
    参数:
        chaos_func: 混沌映射函数
        x0: 初始值 (种子)
        N: 置乱表长度
        M: 暂态迭代轮数 (默认1000)
    返回:
        perm: 置乱表 (0-indexed), perm[i]=j 表示第i个位置映射到第j个位置
    """
    x = x0
    # 暂态迭代
    for _ in range(M):
        x = chaos_func(x, **kwargs)
    
    # 生成混沌序列
    sequence = np.zeros(N)
    for i in range(N):
        x = chaos_func(x, **kwargs)
        sequence[i] = x
    
    # 排序得到置乱索引
    perm = np.argsort(sequence)
    return perm


# ============================================================
# 3. 循环结构分析
# ============================================================

def analyze_cycles(perm):
    """
    分析置乱表的循环结构。
    
    返回:
        cycles: 所有循环的列表 (每个循环是一个元素列表)
        cycle_lengths: 各循环长度列表
        cycle_length_counts: {长度: 出现次数} 的字典
        order: 置乱表的阶 (所有循环长度的LCM)
    """
    N = len(perm)
    visited = np.zeros(N, dtype=bool)
    cycles = []
    
    for i in range(N):
        if visited[i]:
            continue
        cycle = []
        j = i
        while not visited[j]:
            visited[j] = True
            cycle.append(j)
            j = perm[j]
        cycles.append(cycle)
    
    cycle_lengths = [len(c) for c in cycles]
    cycle_length_counts = Counter(cycle_lengths)
    
    # 阶 = LCM(所有循环长度)
    order = reduce(lambda a, b: a * b // gcd(a, b), cycle_lengths)
    
    return cycles, cycle_lengths, cycle_length_counts, order


def compute_average_order(chaos_func, N, num_seeds=50, M=1000, **kwargs):
    """
    对给定N, 使用不同种子计算平均阶。
    """
    np.random.seed(42)
    seeds = np.random.uniform(0.01, 0.99, num_seeds)
    orders = []
    
    for x0 in seeds:
        perm = generate_permutation(chaos_func, x0, N, M, **kwargs)
        _, _, _, order = analyze_cycles(perm)
        orders.append(order)
    
    return orders


# ============================================================
# 4. 详细分析单个置乱表
# ============================================================

def detailed_analysis(chaos_func, func_name, x0, N, M=1000, **kwargs):
    """对单个置乱表进行详细分析并打印结果。"""
    perm = generate_permutation(chaos_func, x0, N, M, **kwargs)
    cycles, cycle_lengths, cycle_length_counts, order = analyze_cycles(perm)
    
    print(f"\n{'='*60}")
    print(f"混沌映射: {func_name}")
    print(f"初始值 x0 = {x0}, 置乱表长度 N = {N}")
    print(f"{'='*60}")
    print(f"循环圈总数: {len(cycles)}")
    print(f"循环长度种类数: {len(cycle_length_counts)}")
    print(f"循环长度分布:")
    for length in sorted(cycle_length_counts.keys()):
        count = cycle_length_counts[length]
        print(f"  长度 {length}: {count} 个循环 (共覆盖 {length * count} 个元素)")
    print(f"总阶 (LCM): {order}")
    print(f"log2(阶): {np.log2(float(order)):.2f}")
    
    return perm, cycles, cycle_lengths, cycle_length_counts, order


# ============================================================
# 5. 实验与绘图
# ============================================================

def experiment_order_vs_N(output_dir="results"):
    """实验: 平均阶 vs N 曲线"""
    os.makedirs(output_dir, exist_ok=True)
    
    maps = [
        (logistic_map, "Logistic Map (μ=3.99)", {"mu": 3.99}),
        (tent_map,     "Tent Map (μ=1.99)",     {"mu": 1.99}),
        (sine_map,     "Sine Map (μ=0.99)",     {"mu": 0.99}),
    ]
    
    N_values = [50, 100, 200, 500, 1000, 2000, 5000]
    num_seeds = 30
    
    plt.figure(figsize=(12, 7))
    
    results = {}
    
    for chaos_func, name, kwargs in maps:
        print(f"\n--- 处理 {name} ---")
        avg_log_orders = []
        std_log_orders = []
        
        for N in N_values:
            print(f"  N={N}...", end=" ", flush=True)
            orders = compute_average_order(chaos_func, N, num_seeds, **kwargs)
            log_orders = [np.log2(float(o)) for o in orders]
            avg_log = np.mean(log_orders)
            std_log = np.std(log_orders)
            avg_log_orders.append(avg_log)
            std_log_orders.append(std_log)
            print(f"avg log2(order)={avg_log:.2f} ± {std_log:.2f}")
        
        results[name] = (avg_log_orders, std_log_orders)
        
        plt.errorbar(N_values, avg_log_orders, yerr=std_log_orders,
                     marker='o', capsize=5, linewidth=2, label=name)
    
    # 理论参考线: 随机置换的期望阶 ≈ exp(sqrt(N * ln(N) / 2))
    # log2 of that ≈ sqrt(N * ln(N) / 2) / ln(2)
    theory_log_orders = [np.sqrt(N * np.log(N) / 2) / np.log(2) for N in N_values]
    plt.plot(N_values, theory_log_orders, '--k', linewidth=1.5,
             label='Random Permutation (theoretical)')
    
    plt.xlabel('Permutation Size N', fontsize=13)
    plt.ylabel('Average log₂(Order)', fontsize=13)
    plt.title('Average Permutation Order vs Size N for Different Chaotic Maps', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/order_vs_N.png', dpi=150)
    plt.close()
    print(f"\n图表已保存: {output_dir}/order_vs_N.png")
    
    return results


def experiment_cycle_distribution(output_dir="results"):
    """实验: 循环长度分布"""
    os.makedirs(output_dir, exist_ok=True)
    
    maps = [
        (logistic_map, "Logistic Map", {"mu": 3.99}),
        (tent_map,     "Tent Map",     {"mu": 1.99}),
        (sine_map,     "Sine Map",     {"mu": 0.99}),
    ]
    
    N = 1000
    x0 = 0.3456
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, (chaos_func, name, kwargs) in enumerate(maps):
        perm = generate_permutation(chaos_func, x0, N, **kwargs)
        _, cycle_lengths, cycle_length_counts, order = analyze_cycles(perm)
        
        lengths = sorted(cycle_length_counts.keys())
        counts = [cycle_length_counts[l] for l in lengths]
        
        axes[idx].bar(lengths, counts, color=['#3274A1', '#E1812C', '#3A923A'][idx], alpha=0.75)
        axes[idx].set_xlabel('Cycle Length', fontsize=12)
        axes[idx].set_ylabel('Count', fontsize=12)
        axes[idx].set_title(f'{name}\nN={N}, Order={order}\n({len(cycle_length_counts)} distinct lengths)',
                           fontsize=11)
        axes[idx].grid(True, alpha=0.3)
    
    plt.suptitle('Cycle Length Distribution for Different Chaotic Maps (x₀=0.3456)', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/cycle_distribution.png', dpi=150)
    plt.close()
    print(f"图表已保存: {output_dir}/cycle_distribution.png")


def experiment_seed_sensitivity(output_dir="results"):
    """实验: 种子敏感性 — 微小种子变化对阶的影响"""
    os.makedirs(output_dir, exist_ok=True)
    
    maps = [
        (logistic_map, "Logistic Map", {"mu": 3.99}),
        (tent_map,     "Tent Map",     {"mu": 1.99}),
        (sine_map,     "Sine Map",     {"mu": 0.99}),
    ]
    
    N = 500
    base_seed = 0.5
    deltas = np.linspace(-0.001, 0.001, 201)
    seeds = base_seed + deltas
    seeds = seeds[(seeds > 0.01) & (seeds < 0.99)]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, (chaos_func, name, kwargs) in enumerate(maps):
        orders = []
        valid_seeds = []
        for s in seeds:
            perm = generate_permutation(chaos_func, s, N, **kwargs)
            _, _, _, order = analyze_cycles(perm)
            orders.append(np.log2(float(order)))
            valid_seeds.append(s - base_seed)
        
        axes[idx].scatter(valid_seeds, orders, s=3, alpha=0.7,
                         color=['#3274A1', '#E1812C', '#3A923A'][idx])
        axes[idx].set_xlabel('Δx₀ (seed perturbation)', fontsize=12)
        axes[idx].set_ylabel('log₂(Order)', fontsize=12)
        axes[idx].set_title(f'{name}\nN={N}, base x₀={base_seed}', fontsize=11)
        axes[idx].grid(True, alpha=0.3)
    
    plt.suptitle('Seed Sensitivity: Order vs Small Perturbation of Initial Value', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/seed_sensitivity.png', dpi=150)
    plt.close()
    print(f"图表已保存: {output_dir}/seed_sensitivity.png")


def experiment_parameter_sensitivity(output_dir="results"):
    """实验: 参数敏感性 — 不同参数μ对平均阶的影响"""
    os.makedirs(output_dir, exist_ok=True)
    
    N = 500
    num_seeds = 20
    np.random.seed(123)
    seeds = np.random.uniform(0.01, 0.99, num_seeds)
    
    # Logistic: mu in (3.57, 4)
    mus_logistic = np.linspace(3.6, 3.999, 50)
    avg_orders_logistic = []
    for mu in mus_logistic:
        orders = []
        for x0 in seeds:
            perm = generate_permutation(logistic_map, x0, N, mu=mu)
            _, _, _, order = analyze_cycles(perm)
            orders.append(np.log2(float(order)))
        avg_orders_logistic.append(np.mean(orders))
    
    # Tent: mu in (1, 2)
    mus_tent = np.linspace(1.1, 1.999, 50)
    avg_orders_tent = []
    for mu in mus_tent:
        orders = []
        for x0 in seeds:
            perm = generate_permutation(tent_map, x0, N, mu=mu)
            _, _, _, order = analyze_cycles(perm)
            orders.append(np.log2(float(order)))
        avg_orders_tent.append(np.mean(orders))
    
    # Sine: mu in (0, 1)
    mus_sine = np.linspace(0.8, 0.999, 50)
    avg_orders_sine = []
    for mu in mus_sine:
        orders = []
        for x0 in seeds:
            perm = generate_permutation(sine_map, x0, N, mu=mu)
            _, _, _, order = analyze_cycles(perm)
            orders.append(np.log2(float(order)))
        avg_orders_sine.append(np.mean(orders))
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    axes[0].plot(mus_logistic, avg_orders_logistic, 'o-', color='#3274A1', markersize=3)
    axes[0].set_xlabel('μ', fontsize=13)
    axes[0].set_ylabel('Avg log₂(Order)', fontsize=12)
    axes[0].set_title('Logistic Map', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(mus_tent, avg_orders_tent, 'o-', color='#E1812C', markersize=3)
    axes[1].set_xlabel('μ', fontsize=13)
    axes[1].set_ylabel('Avg log₂(Order)', fontsize=12)
    axes[1].set_title('Tent Map', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(mus_sine, avg_orders_sine, 'o-', color='#3A923A', markersize=3)
    axes[2].set_xlabel('μ', fontsize=13)
    axes[2].set_ylabel('Avg log₂(Order)', fontsize=12)
    axes[2].set_title('Sine Map', fontsize=12)
    axes[2].grid(True, alpha=0.3)
    
    plt.suptitle(f'Parameter Sensitivity: Avg Order vs μ (N={N})', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/parameter_sensitivity.png', dpi=150)
    plt.close()
    print(f"图表已保存: {output_dir}/parameter_sensitivity.png")


def experiment_permutation_quality(output_dir="results"):
    """实验: 置乱质量评估 — 偏移距离分布"""
    os.makedirs(output_dir, exist_ok=True)
    
    maps = [
        (logistic_map, "Logistic Map", {"mu": 3.99}),
        (tent_map,     "Tent Map",     {"mu": 1.99}),
        (sine_map,     "Sine Map",     {"mu": 0.99}),
    ]
    
    N = 1000
    x0 = 0.3456
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, (chaos_func, name, kwargs) in enumerate(maps):
        perm = generate_permutation(chaos_func, x0, N, **kwargs)
        displacements = np.abs(perm - np.arange(N))
        
        axes[idx].hist(displacements, bins=50, density=True,
                      color=['#3274A1', '#E1812C', '#3A923A'][idx], alpha=0.75)
        axes[idx].set_xlabel('|perm(i) - i|', fontsize=12)
        axes[idx].set_ylabel('Probability Density', fontsize=12)
        axes[idx].set_title(f'{name}\nMean disp={np.mean(displacements):.1f}, '
                           f'Fixed pts={np.sum(displacements==0)}',
                           fontsize=11)
        axes[idx].grid(True, alpha=0.3)
    
    plt.suptitle('Displacement Distribution (Permutation Quality)', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/displacement_distribution.png', dpi=150)
    plt.close()
    print(f"图表已保存: {output_dir}/displacement_distribution.png")


# ============================================================
# 6. 主程序
# ============================================================

if __name__ == "__main__":
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("混沌置乱的循环阶分析")
    print("Cycle Order Analysis of Chaotic Permutations")
    print("=" * 60)
    
    # 详细分析示例
    print("\n\n>>> 1. 单个置乱表详细分析 (N=100)")
    for func, name, kwargs in [
        (logistic_map, "Logistic Map (μ=3.99)", {"mu": 3.99}),
        (tent_map,     "Tent Map (μ=1.99)",     {"mu": 1.99}),
        (sine_map,     "Sine Map (μ=0.99)",     {"mu": 0.99}),
    ]:
        detailed_analysis(func, name, 0.3456, 100, **kwargs)
    
    print("\n\n>>> 2. 单个置乱表详细分析 (N=1000)")
    for func, name, kwargs in [
        (logistic_map, "Logistic Map (μ=3.99)", {"mu": 3.99}),
        (tent_map,     "Tent Map (μ=1.99)",     {"mu": 1.99}),
        (sine_map,     "Sine Map (μ=0.99)",     {"mu": 0.99}),
    ]:
        detailed_analysis(func, name, 0.3456, 1000, **kwargs)
    
    # 实验
    print("\n\n>>> 3. 实验: 平均阶 vs N")
    t0 = time.time()
    experiment_order_vs_N(output_dir)
    print(f"用时: {time.time()-t0:.1f}s")
    
    print("\n\n>>> 4. 实验: 循环长度分布")
    experiment_cycle_distribution(output_dir)
    
    print("\n\n>>> 5. 实验: 种子敏感性")
    experiment_seed_sensitivity(output_dir)
    
    print("\n\n>>> 6. 实验: 参数敏感性")
    t0 = time.time()
    experiment_parameter_sensitivity(output_dir)
    print(f"用时: {time.time()-t0:.1f}s")
    
    print("\n\n>>> 7. 实验: 置乱质量评估")
    experiment_permutation_quality(output_dir)
    
    print("\n\n所有实验完成！结果保存在", output_dir, "目录")
