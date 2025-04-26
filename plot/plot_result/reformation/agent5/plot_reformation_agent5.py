import os
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator

# 定义平滑函数
def smoother(x, a=0.9, w=1, mode="window"):
    if mode == "window":
        y = []
        for i in range(len(x)):
            y.append(np.mean(x[max(i - w, 0):i + 1]))
    elif mode == "moving":
        y = [x[0]]
        for i in range(1, len(x)):
            y.append((1 - a) * x[i] + a * y[i - 1])
    else:
        raise NotImplementedError
    return y

# 处理单个种子的数据并应用平滑
def process_single_seed_data(data, window_size=80, scale=1.0):
    # 应用平滑
    smoothed_data = smoother(np.asarray(data), w=window_size, mode="window")
    x = np.arange(0, len(smoothed_data)) * scale
    return x, smoothed_data

# 从tensorboard日志中读取数据
def read_tensorboard_data(log_dir, tag):
    try:
        ea = event_accumulator.EventAccumulator(
            log_dir,
            size_guidance={
                event_accumulator.SCALARS: 0,  # 加载所有scalars
            }
        )
        ea.Reload()
        
        if tag not in ea.scalars.Keys():
            print(f"找不到标签 {tag} 在 {log_dir}")
            return None
        
        # 获取数据并转换为DataFrame
        events = ea.Scalars(tag)
        data = pd.DataFrame([(e.step, e.value) for e in events], 
                           columns=['step', 'value'])
        return data
    except Exception as e:
        print(f"处理 {log_dir} 时出错: {e}")
        return None

# 主函数
def plot_heading_results():
    # 定义日志目录路径和对应的种子
    log_dirs = [
        "/home/dqy/NeuralPlanex/AeroPlanex_v/AeroPlanax/plot/reformation policy(agent 5 seed 0)/2025-04-24-02-32/logs",
        "/home/dqy/NeuralPlanex/AeroPlanex_v/AeroPlanax/plot/reformation policy(agent 5 seed 10)/2025-04-24-02-34/logs",
        "/home/dqy/NeuralPlanex/AeroPlanex_v/AeroPlanax/plot/reformation policy(agent 5 seed 42)/2025-04-24-02-41/logs"
    ]
    seeds = [0, 10, 42]
    
    # 只绘制回合奖励指标
    metric = "eval/episodic_return"
    
    # 设置绘图风格
    sns.set_theme(
        style="darkgrid",
        font_scale=1.2,
        rc={"figure.figsize": (10, 6)}
    )
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    colors = sns.color_palette("Set1", 3)
    
    # 为每个种子读取和处理数据
    for i, (log_dir, seed) in enumerate(zip(log_dirs, seeds)):
        print(f"\n处理种子 {seed} 的数据")
        
        # 读取数据
        data = read_tensorboard_data(log_dir, metric)
        if data is None or data.empty:
            print(f"警告: 无法读取种子 {seed} 的数据")
            continue
        
        # 处理数据 - 使用窗口平滑
        x, y = process_single_seed_data(data['value'].values, window_size=80, scale=300*3000/1e6)
        
        # 计算本地方差 (使用滑动窗口)
        window_size = 10  # 本地方差窗口大小
        local_std = []
        for j in range(len(y)):
            window_start = max(0, j - window_size)
            window_end = min(len(y), j + window_size + 1)
            local_std.append(np.std(y[window_start:window_end]))
        local_std = np.array(local_std)
        
        # 绘制曲线和本地方差阴影
        ax.plot(x, y, color=colors[i], label=f"Seed {seed}", linewidth=2)
        ax.fill_between(x, y - local_std, y + local_std, color=colors[i], alpha=0.2)
        
        # 输出最终值
        print(f"种子 {seed} 最终值: {y[-1]:.2f} ± {local_std[-1]:.2f}")
    
    # 设置图表标题和标签
    ax.set_title("reformation Policy Training Performance(agent5)", fontsize=14)
    ax.set_xlabel("Million Environment Steps", fontsize=12)
    ax.set_ylabel("Average Episode Return", fontsize=12)
    ax.legend(loc="best", fontsize=12)
    
    # 显示网格
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # 调整布局并保存
    plt.tight_layout()
    plt.savefig("/home/dqy/NeuralPlanex/AeroPlanex_v/AeroPlanax/plot/plot_result/reformation/agent5/reformation_policy_return_by_seed(agent5).png", bbox_inches="tight")
    plt.savefig("/home/dqy/NeuralPlanex/AeroPlanex_v/AeroPlanax/plot/plot_result/reformation/agent5/reformation_policy_return_by_seed(agent5).pdf", bbox_inches="tight")
    print("\n图表已保存为 reformation_policy_return_by_seed(agent5).png 和 reformation_policy_return_by_seed(agent5).pdf")
    plt.show()

# 执行主函数
if __name__ == "__main__":
    plot_heading_results()