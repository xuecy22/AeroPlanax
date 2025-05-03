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

# 处理单个数据集并应用平滑
def process_single_run_data(data, window_size=80, scale=1.0):
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

def plot_combat_performance():
    # 配置不同规模和类型的实验路径
    experiments = {
        # 1v1配置
        "1v1 Selfplay": [
            "/home/dqy/aeroplanax/AeroPlanax_f16/results/多机对战（1v1 selfplay 底层heading policy）/2025-05-01-01-35/logs",
        ],
        "1v1 vs Baseline": [
            "/home/dqy/aeroplanax/AeroPlanax_f16/results/多机对战（1v1 vsbaseline 底层heading policy）/2025-05-01-02-01/logs",
        ],
        
        # 2v2配置
        "2v2 Selfplay": [
            "/home/dqy/aeroplanax/AeroPlanax_f16/results/多机对战（2v2 selfplay 底层heading policy）/2025-05-01-01-37/logs",
        ],
        "2v2 vs Baseline": [
            "/home/dqy/aeroplanax/AeroPlanax_f16/results/多机对战（2v2 vsbaseline 底层heading policy）/2025-05-01-02-05/logs",
        ],
        
        # 5v5配置
        "5v5 Selfplay": [
            "/home/dqy/aeroplanax/AeroPlanax_f16/results/多机对战（5v5 selfplay 底层heading policy）/2025-05-01-01-59/logs",
        ],
        "5v5 vs Baseline": [
            "/home/dqy/aeroplanax/AeroPlanax_f16/results/多机对战（5v5 vsbaseline 底层heading policy）/2025-05-01-01-57/logs",
        ],
        
        # 10v10配置
        "10v10 Selfplay": [
            "/home/dqy/aeroplanax/AeroPlanax_f16/results/多机对战（10v10 selfplay 底层heading policy）/2025-05-01-01-43/logs",
        ],
        "10v10 vs Baseline": [
            "/home/dqy/aeroplanax/AeroPlanax_f16/results/多机对战（10v10 vsbaseline 底层heading policy）/2025-05-01-01-56/logs",
        ],
    }
    
    # 要绘制的指标
    metric = "eval/episodic_return"
    
    # 设置绘图风格
    sns.set_theme(
        style="darkgrid",
        font_scale=1.2,
        rc={"figure.figsize": (12, 8)}
    )
    
    # 创建两个子图，一个用于selfplay，一个用于vsbaseline
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), dpi=300)
    
    # 为不同规模选择不同颜色
    scales = ["1v1", "2v2", "5v5", "10v10"]
    colors = sns.color_palette("viridis", len(scales))
    
    # 创建颜色映射
    color_map = {scale: color for scale, color in zip(scales, colors)}
    
    # 存储最终性能数据以便后续总结
    final_performance = {}
    
    # 处理每个实验
    for exp_name, log_dirs in experiments.items():
        print(f"\n处理 {exp_name} 数据")
        
        # 确定使用哪个子图
        ax = ax1 if "Selfplay" in exp_name else ax2
        
        # 获取对战规模信息用于颜色分配
        scale = exp_name.split(" ")[0]  # 例如 "1v1", "2v2"
        color = color_map[scale]
        
        # 存储所有数据
        all_curves = []
        min_length = float('inf')
        
        # 处理每个日志目录
        for log_dir in log_dirs:
            data = read_tensorboard_data(log_dir, metric)
            if data is None or data.empty:
                print(f"警告: 无法读取 {log_dir} 的数据")
                continue
            
            # 平滑处理
            _, smoothed_data = process_single_run_data(data['value'].values, window_size=80)
            all_curves.append(smoothed_data)
            
            # 记录最短曲线长度
            min_length = min(min_length, len(smoothed_data))
        
        # 如果没有有效数据，跳过此实验
        if not all_curves:
            print(f"跳过 {exp_name}: 没有有效数据")
            continue
            
        # 截断所有曲线到相同长度
        all_curves = [curve[:min_length] for curve in all_curves]
        
        # 转换为numpy数组并计算统计量
        all_curves = np.array(all_curves)
        mean_curve = np.mean(all_curves, axis=0)
        std_curve = np.std(all_curves, axis=0)
        
        # 对于单个曲线，std可能为0，设置为小值以避免绘图问题
        if len(all_curves) == 1:
            std_curve = np.ones_like(std_curve) * 0.1
        
        # 创建x轴(环境步数，单位为百万)
        # 注意：这里可能需要根据您的实际训练步数调整
        x = np.arange(0, min_length) * (300*1000/1e6)  # 假设每300k步记录一次
        
        # 绘制均值曲线
        ax.plot(x, mean_curve, color=color, label=exp_name, linewidth=2)
        
        # 添加标准差区域
        ax.fill_between(x, mean_curve - std_curve, mean_curve + std_curve, 
                        color=color, alpha=0.2)
        
        # 记录最终性能
        final_perf = mean_curve[-1]
        final_std = std_curve[-1]
        final_performance[exp_name] = (final_perf, final_std)
        print(f"{exp_name} 最终性能: {final_perf:.2f} ± {final_std:.2f}")
    
    # 设置子图标题和标签
    ax1.set_title("Self-play Training", fontsize=16)
    ax1.set_xlabel("Million Environment Steps", fontsize=14)
    ax1.set_ylabel("Average Episode Return", fontsize=14)
    ax1.legend(loc="best", fontsize=12)
    
    ax2.set_title("Training vs Baseline", fontsize=16)
    ax2.set_xlabel("Million Environment Steps", fontsize=14)
    ax2.set_ylabel("Average Episode Return", fontsize=14)
    ax2.legend(loc="best", fontsize=12)
    
    # 确保两个子图使用相同的y轴范围以便比较
    y_min = min(ax1.get_ylim()[0], ax2.get_ylim()[0])
    y_max = max(ax1.get_ylim()[1], ax2.get_ylim()[1])
    ax1.set_ylim(y_min, y_max)
    ax2.set_ylim(y_min, y_max)
    
    # 添加网格
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # 调整布局并保存
    plt.tight_layout()
    output_dir = "/home/dqy/aeroplanax/AeroPlanax_f16/plot/plot_result/combat/"
    os.makedirs(output_dir, exist_ok=True)
    
    plt.savefig(f"{output_dir}combat_training_comparison.png", bbox_inches="tight")
    plt.savefig(f"{output_dir}combat_training_comparison.pdf", bbox_inches="tight")
    print(f"\n图表已保存为 {output_dir}combat_training_comparison.png 和 .pdf")
    
    # 输出最终性能总结
    print("\n最终性能总结:")
    for exp_name, (perf, std) in final_performance.items():
        print(f"{exp_name}: {perf:.2f} ± {std:.2f}")
    
    plt.show()

if __name__ == "__main__":
    plot_combat_performance()