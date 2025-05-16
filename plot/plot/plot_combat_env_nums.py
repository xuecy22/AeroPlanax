import os
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
from tensorboard.backend.event_processing import event_accumulator

# 设置字体为英文
plt.rcParams['font.family'] = ['Arial']
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
            print(f"Tag {tag} not found in {log_dir}")
            return None
        
        # 获取数据并转换为DataFrame
        events = ea.Scalars(tag)
        data = pd.DataFrame([(e.step, e.value) for e in events], 
                           columns=['step', 'value'])
        return data
    except Exception as e:
        print(f"Error processing {log_dir}: {e}")
        return None

def plot_combat_performance():
    # 配置不同规模的实验路径，按规模分组
    experiments_by_scale = {
        "ENV_NUM : 30": {
            "Seed 0": ["/home/dqy/aeroplanax/AeroPlanax_heading/plot/plot_result/new/combat/different NUM ENVS/combat_agent2_vsbaseline_NUM_ENVS_30_seed0/logs"],
            "Seed 10": ["/home/dqy/aeroplanax/AeroPlanax_heading/plot/plot_result/new/combat/different NUM ENVS/combat_agent2_vsbaseline_NUM_ENVS_30_seed10/logs"],
            "Seed 20": ["/home/dqy/aeroplanax/AeroPlanax_heading/plot/plot_result/new/combat/different NUM ENVS/combat_agent2_vsbaseline_NUM_ENVS_30_seed20/logs"],
            "Seed 30": ["/home/dqy/aeroplanax/AeroPlanax_heading/plot/plot_result/new/combat/different NUM ENVS/combat_agent2_vsbaseline_NUM_ENVS_30_seed30/logs"],
            "Seed 42": ["/home/dqy/aeroplanax/AeroPlanax_heading/plot/plot_result/new/combat/different NUM ENVS/combat_agent2_vsbaseline_NUM_ENVS_30_seed42/logs"],
        },
        "ENV_NUM : 300": {
            "Seed 0": ["/home/dqy/aeroplanax/AeroPlanax_heading/plot/plot_result/new/combat/different NUM ENVS/combat_agent2_vsbaseline_NUM_ENVS_300_seed0/logs"],
            "Seed 10": ["/home/dqy/aeroplanax/AeroPlanax_heading/plot/plot_result/new/combat/different NUM ENVS/combat_agent2_vsbaseline_NUM_ENVS_300_seed10/logs"],
            "Seed 20": ["/home/dqy/aeroplanax/AeroPlanax_heading/plot/plot_result/new/combat/different NUM ENVS/combat_agent2_vsbaseline_NUM_ENVS_300_seed20/logs"],
            "Seed 30": ["/home/dqy/aeroplanax/AeroPlanax_heading/plot/plot_result/new/combat/different NUM ENVS/combat_agent2_vsbaseline_NUM_ENVS_300_seed30/logs"],
            "Seed 42": ["/home/dqy/aeroplanax/AeroPlanax_heading/plot/plot_result/new/combat/different NUM ENVS/combat_agent2_vsbaseline_NUM_ENVS_300_seed42/logs"],
        },
        "ENV_NUM : 3000": {
            "Seed 0": ["/home/dqy/aeroplanax/AeroPlanax_heading/plot/plot_result/new/combat/different NUM ENVS/combat_agent2_vsbaseline_NUM_ENVS_3000_seed0/logs"],
            "Seed 10": ["/home/dqy/aeroplanax/AeroPlanax_heading/plot/plot_result/new/combat/different NUM ENVS/combat_agent2_vsbaseline_NUM_ENVS_3000_seed10/logs"],
            "Seed 20": ["/home/dqy/aeroplanax/AeroPlanax_heading/plot/plot_result/new/combat/different NUM ENVS/combat_agent2_vsbaseline_NUM_ENVS_3000_seed20/logs"],
            "Seed 30": ["/home/dqy/aeroplanax/AeroPlanax_heading/plot/plot_result/new/combat/different NUM ENVS/combat_agent2_vsbaseline_NUM_ENVS_3000_seed42/logs"],
            "Seed 42": ["/home/dqy/aeroplanax/AeroPlanax_heading/plot/plot_result/new/combat/different NUM ENVS/combat_agent2_vsbaseline_NUM_ENVS_3000_seed42/logs"],
        },
    }
    
    # 这里所有数据都是1v1（agent2表示总共2个智能体），所以不需要缩放
    scale_factor = 1.0  # 设置为1表示保持原值
    
    # 要绘制的指标
    metric = "eval/episodic_return"
    
    # 设置绘图风格
    sns.set_theme(
        style="darkgrid",
        font_scale=1.2,
        rc={"figure.figsize": (12, 8)}
    )
    
    # 创建单个图表
    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
    
    # 设置红绿蓝颜色
    colors = ['red', 'green', 'dodgerblue']
    scales = ["ENV_NUM : 30", "ENV_NUM : 300", "ENV_NUM : 3000"]
    
    # 创建颜色映射
    color_map = {scale: color for scale, color in zip(scales, colors)}
    
    # 存储最终性能数据以便后续总结
    final_performance = {}
    
    # 存储所有实验的曲线长度，用于找到全局最短长度
    all_experiments_min_lengths = []
    all_processed_data = {}
    
    # 第一遍：处理每个规模的实验，找到最短长度
    for scale, seeds_data in experiments_by_scale.items():
        print(f"\nProcessing {scale} scale data")
        
        # 存储这个规模下所有种子的曲线
        all_scale_curves = []
        scale_min_length = float('inf')
        
        # 处理每个种子的数据
        for seed_name, log_dirs in seeds_data.items():
            # 处理每个日志目录
            for log_dir in log_dirs:
                data = read_tensorboard_data(log_dir, metric)
                if data is None or data.empty:
                    print(f"Warning: Cannot read data from {log_dir}")
                    continue
                
                # 平滑处理
                _, smoothed_data = process_single_run_data(data['value'].values, window_size=5)
                all_scale_curves.append(smoothed_data)
                
                # 记录当前规模的最短曲线长度
                scale_min_length = min(scale_min_length, len(smoothed_data))
        
        # 如果没有有效数据，跳过此规模
        if not all_scale_curves:
            print(f"Skipping {scale}: No valid data")
            continue
        
        # 记录这个规模的最短长度和处理后的数据
        all_experiments_min_lengths.append(scale_min_length)
        all_processed_data[scale] = all_scale_curves
    
    # 找到所有实验中的最短长度
    if not all_experiments_min_lengths:
        print("No valid data found for any scale")
        return
    
    global_min_length = min(all_experiments_min_lengths)
    print(f"\nGlobal minimum length across all experiments: {global_min_length}")
    
    # 第二遍：使用全局最短长度重新绘制所有曲线
    for scale, all_scale_curves in all_processed_data.items():
        # 截断所有曲线到全局最短长度
        all_scale_curves = [curve[:global_min_length] for curve in all_scale_curves]
        
        # 转换为numpy数组并计算统计量
        all_scale_curves = np.array(all_scale_curves)
        mean_curve = np.mean(all_scale_curves, axis=0)
        std_curve = np.std(all_scale_curves, axis=0)
        
        # 应用缩放（在这种情况下不需要缩放）
        scaled_mean_curve = mean_curve * scale_factor
        scaled_std_curve = std_curve * scale_factor
        
        # 创建x轴(环境步数，单位为百万)
        x = np.arange(0, global_min_length) * (300*1000/1e6)  # 假设每300k步记录一次
        
        # 绘制均值曲线
        ax.plot(x, scaled_mean_curve, color=color_map[scale], label=f"{scale}", linewidth=2)
        
        # 添加标准差区域
        ax.fill_between(x, scaled_mean_curve - scaled_std_curve, scaled_mean_curve + scaled_std_curve, 
                       color=color_map[scale], alpha=0.2)
        
        # 记录最终性能
        final_perf = scaled_mean_curve[-1]
        final_std = scaled_std_curve[-1]
        final_performance[scale] = (final_perf, final_std)
        print(f"{scale} final performance: {final_perf:.2f} ± {final_std:.2f}")
    
    # 设置图表标题和标签
    ax.set_title("Combat Task (Different NUM ENVS) Training Performance", fontsize=16)
    ax.set_xlabel("Million Environment Steps", fontsize=14)
    ax.set_ylabel("Average Reward", fontsize=14)
    ax.legend(loc="lower right", fontsize=12)
    
    # 添加网格
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # 调整布局并保存
    plt.tight_layout()
    output_dir = "/home/dqy/aeroplanax/AeroPlanax_heading/plot/plot_result/new/combat/different NUM ENVS/"
    os.makedirs(output_dir, exist_ok=True)
    
    plt.savefig(f"{output_dir}Combat_Different_NUM_ENVS.png", bbox_inches="tight")
    plt.savefig(f"{output_dir}Combat_Different_NUM_ENVS.pdf", bbox_inches="tight")
    print(f"\nPlots saved as {output_dir}Combat_Different_NUM_ENVS.png and .pdf")
    
    # 输出最终性能总结
    print("\nFinal Performance Summary:")
    for scale, (perf, std) in final_performance.items():
        print(f"{scale}: {perf:.2f} ± {std:.2f}")
    
    plt.show()

if __name__ == "__main__":
    plot_combat_performance()