import os
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
from tensorboard.backend.event_processing import event_accumulator

# 设置英文字体
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

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

# 处理单个数据集并应用平滑
def process_single_run_data(data, window_size=20, scale=1.0):
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
                event_accumulator.SCALARS: 0,
            }
        )
        ea.Reload()
        
        if tag not in ea.scalars.Keys():
            print(f"Could not find tag {tag} in {log_dir}")
            return None
        
        # 获取数据并转换为DataFrame
        events = ea.Scalars(tag)
        data = pd.DataFrame([(e.step, e.value) for e in events], 
                           columns=['step', 'value'])
        return data
    except Exception as e:
        print(f"Error processing {log_dir}: {e}")
        return None

def plot_combat_comparison():
    # 配置不同规模的实验路径 - End to End
    end_to_end_experiments = {
        "1v1": {
            "Seed 0": ["/home/dqy/aeroplanax/AeroPlanax_heading/plot/plot_result/combat_selfplay_new/end to end/combat_end_to_end_agent2_selfplay_seed0/logs"],
            "Seed 10": ["/home/dqy/aeroplanax/AeroPlanax_heading/plot/plot_result/combat_selfplay_new/end to end/combat_end_to_end_agent2_selfplay_seed10/logs"],
            "Seed 42": ["/home/dqy/aeroplanax/AeroPlanax_heading/plot/plot_result/combat_selfplay_new/end to end/combat_end_to_end_agent2_selfplay_seed42/logs"],
        },
        "2v2": {
            "Seed 0": ["/home/dqy/aeroplanax/AeroPlanax_heading/plot/plot_result/combat_selfplay_new/end to end/combat_end_to_end_agent4_selfplay_seed0/logs"],
            "Seed 10": ["/home/dqy/aeroplanax/AeroPlanax_heading/plot/plot_result/combat_selfplay_new/end to end/combat_end_to_end_agent4_selfplay_seed10/logs"],
            "Seed 42": ["/home/dqy/aeroplanax/AeroPlanax_heading/plot/plot_result/combat_selfplay_new/end to end/combat_end_to_end_agent4_selfplay_seed42/logs"],
        },
        "5v5": {
            "Seed 0": ["/home/dqy/aeroplanax/AeroPlanax_heading/plot/plot_result/combat_selfplay_new/end to end/combat_end_to_end_agent10_selfplay_seed0/logs"],
            "Seed 10": ["/home/dqy/aeroplanax/AeroPlanax_heading/plot/plot_result/combat_selfplay_new/end to end/combat_end_to_end_agent10_selfplay_seed10/logs"],
            "Seed 42": ["/home/dqy/aeroplanax/AeroPlanax_heading/plot/plot_result/combat_selfplay_new/end to end/combat_end_to_end_agent10_selfplay_seed42/logs"],
        },
    }
    
    # 配置不同规模的实验路径 - 分层 Selfplay
    hierarchical_experiments = {
        "1v1": {
            "Seed 0": ["/home/dqy/aeroplanax/AeroPlanax_heading/plot/plot_result/combat_selfplay_new/combat_agent_2_selfplay_seed0/logs"],
            "Seed 10": ["/home/dqy/aeroplanax/AeroPlanax_heading/plot/plot_result/combat_selfplay_new/combat_agent2_selfplay_seed10/logs"],
            "Seed 42": ["/home/dqy/aeroplanax/AeroPlanax_heading/plot/plot_result/combat_selfplay_new/combat_agent2_selfplay_seed42/logs"],
        },
        "2v2": {
            "Seed 0": ["/home/dqy/aeroplanax/AeroPlanax_heading/plot/plot_result/combat_selfplay_new/combat_agent4_selfplay_seed0/logs"],
            "Seed 10": ["/home/dqy/aeroplanax/AeroPlanax_heading/plot/plot_result/combat_selfplay_new/combat_agent4_selfplay_seed10/logs"],
            "Seed 42": ["/home/dqy/aeroplanax/AeroPlanax_heading/plot/plot_result/combat_selfplay_new/combat_agent4_selfplay_seed42/logs"],
        },
        "5v5": {
            "Seed 0": ["/home/dqy/aeroplanax/AeroPlanax_heading/plot/plot_result/combat_selfplay_new/combat_agent10_selfplay_seed0/logs"],
            "Seed 10": ["/home/dqy/aeroplanax/AeroPlanax_heading/plot/plot_result/combat_selfplay_new/combat_agent10_selfplay_seed10/logs"],
            "Seed 42": ["/home/dqy/aeroplanax/AeroPlanax_heading/plot/plot_result/combat_selfplay_new/combat_agent10_selfplay_seed42/logs"],
        },
    }
    
    # 要绘制的指标
    metric = "eval/episodic_return"
    
    # 设置绘图风格
    sns.set_theme(
        style="darkgrid",
        font_scale=1.2,
    )
    
    # 创建一行三列的子图
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=300, sharey=False)
    
    # 设置方法的颜色
    method_colors = {
        "End-to-End": "red",
        "Hierarchical": "blue"
    }
    
    # 存储所有子图的y轴范围，用于后续统一
    y_ranges = []
    
    # 第一轮循环：处理数据并计算每个图的范围
    for i, scale in enumerate(["1v1", "2v2", "5v5"]):
        e2e_data = end_to_end_experiments[scale]
        hier_data = hierarchical_experiments[scale]
        
        ax = axes[i]
        
        # 处理End-to-End数据
        e2e_curves = []
        e2e_min_length = float('inf')
        
        for seed_name, log_dirs in e2e_data.items():
            for log_dir in log_dirs:
                data = read_tensorboard_data(log_dir, metric)
                if data is None or data.empty:
                    continue
                
                _, smoothed_data = process_single_run_data(data['value'].values, window_size=1)
                e2e_curves.append(smoothed_data)
                e2e_min_length = min(e2e_min_length, len(smoothed_data))
        
        # 处理分层数据
        hier_curves = []
        hier_min_length = float('inf')
        
        for seed_name, log_dirs in hier_data.items():
            for log_dir in log_dirs:
                data = read_tensorboard_data(log_dir, metric)
                if data is None or data.empty:
                    continue
                
                _, smoothed_data = process_single_run_data(data['value'].values, window_size=1)
                hier_curves.append(smoothed_data)
                hier_min_length = min(hier_min_length, len(smoothed_data))
        
        # 确保两种方法有数据
        if not e2e_curves or not hier_curves:
            continue
            
        # 确定最短曲线的长度
        common_min_length = min(e2e_min_length, hier_min_length)
        
        # 截断到相同长度
        e2e_curves = [curve[:common_min_length] for curve in e2e_curves]
        hier_curves = [curve[:common_min_length] for curve in hier_curves]
        
        # 计算统计量
        e2e_curves = np.array(e2e_curves)
        e2e_mean = np.mean(e2e_curves, axis=0)
        e2e_std = np.std(e2e_curves, axis=0)
        
        hier_curves = np.array(hier_curves)
        hier_mean = np.mean(hier_curves, axis=0)
        hier_std = np.std(hier_curves, axis=0)
        
        # 确定y轴范围
        min_val = min(np.min(e2e_mean - e2e_std), np.min(hier_mean - hier_std))
        max_val = max(np.max(e2e_mean + e2e_std), np.max(hier_mean + hier_std))
        
        # 添加一些边距
        range_size = max_val - min_val
        min_val = min_val - 0.1 * range_size
        max_val = max_val + 0.1 * range_size
        
        y_ranges.append((min_val, max_val, common_min_length))
    
    # 计算所有子图中的最大y轴范围
    max_range = 0
    for y_min, y_max, _ in y_ranges:
        range_size = y_max - y_min
        max_range = max(max_range, range_size)
    
    # 调整y轴范围，保持中心点不变，但范围大小统一
    adjusted_y_ranges = []
    for y_min, y_max, common_min_length in y_ranges:
        center = (y_max + y_min) / 2
        half_range = max_range / 2
        adjusted_y_ranges.append((center - half_range, center + half_range, common_min_length))
    
    # 第二轮循环：实际绘制图形
    for i, scale in enumerate(["1v1", "2v2", "5v5"]):
        e2e_data = end_to_end_experiments[scale]
        hier_data = hierarchical_experiments[scale]
        
        ax = axes[i]
        
        # 获取调整后的范围和最短长度
        y_min, y_max, common_min_length = adjusted_y_ranges[i]
        
        # 处理End-to-End数据
        e2e_curves = []
        
        for seed_name, log_dirs in e2e_data.items():
            for log_dir in log_dirs:
                data = read_tensorboard_data(log_dir, metric)
                if data is None or data.empty:
                    continue
                
                _, smoothed_data = process_single_run_data(data['value'].values, window_size=1)
                e2e_curves.append(smoothed_data[:common_min_length])  # 直接截断到共同长度
        
        # 处理分层数据
        hier_curves = []
        
        for seed_name, log_dirs in hier_data.items():
            for log_dir in log_dirs:
                data = read_tensorboard_data(log_dir, metric)
                if data is None or data.empty:
                    continue
                
                _, smoothed_data = process_single_run_data(data['value'].values, window_size=1)
                hier_curves.append(smoothed_data[:common_min_length])  # 直接截断到共同长度
        
        # 确保两种方法有数据
        if not e2e_curves or not hier_curves:
            continue
            
        # 计算统计量
        e2e_curves = np.array(e2e_curves)
        e2e_mean = np.mean(e2e_curves, axis=0)
        e2e_std = np.std(e2e_curves, axis=0)
        
        hier_curves = np.array(hier_curves)
        hier_mean = np.mean(hier_curves, axis=0)
        hier_std = np.std(hier_curves, axis=0)
        
        # 创建x轴 - 使用相同长度
        x = np.arange(0, common_min_length) * (300*1000/1e6)
        
        # 绘制End-to-End曲线
        ax.plot(x, e2e_mean, color=method_colors["End-to-End"], label="End-to-End", linewidth=2)
        ax.fill_between(x, e2e_mean - e2e_std, e2e_mean + e2e_std, 
                        color=method_colors["End-to-End"], alpha=0.2)
        
        # 绘制分层曲线
        ax.plot(x, hier_mean, color=method_colors["Hierarchical"], label="Hierarchical", linewidth=2)
        ax.fill_between(x, hier_mean - hier_std, hier_mean + hier_std, 
                        color=method_colors["Hierarchical"], alpha=0.2)
        
        # 设置统一的y轴范围
        ax.set_ylim(y_min, y_max)
        
        # 设置标题和标签
        ax.set_title(f"{scale} Combat Task", fontsize=14)
        ax.set_xlabel("Million Environment Steps", fontsize=12)
        if i == 0:  # 只在第一个图上添加y轴标签
            ax.set_ylabel("Average Reward", fontsize=12)
        
        # 在每个图上添加图例
        ax.legend(loc="lower right", fontsize=10)
        
        # 添加网格
        ax.grid(True, linestyle='--', alpha=0.7)
    
    # 设置整体标题
    fig.suptitle("Combat Task Training Performance: End-to-End vs. Hierarchical", fontsize=16, y=1.05)
    
    # 调整布局并保存
    plt.tight_layout()
    output_dir = "/home/dqy/aeroplanax/AeroPlanax_heading/plot/plot_result/new/combat/comparison/"
    os.makedirs(output_dir, exist_ok=True)
    
    plt.savefig(f"{output_dir}combat_e2e_vs_hierarchical.png", bbox_inches="tight")
    plt.savefig(f"{output_dir}combat_e2e_vs_hierarchical.pdf", bbox_inches="tight")
    print(f"Comparison plot saved as {output_dir}combat_e2e_vs_hierarchical.png and .pdf")
    
    plt.show()

if __name__ == "__main__":
    plot_combat_comparison()